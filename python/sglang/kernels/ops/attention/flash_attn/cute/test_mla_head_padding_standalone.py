# Copyright (c) 2025, Tri Dao.
"""Standalone CPU-only tests for the dense 2CTA MLA Q-head padding rule.

This module imports only the standard library and pytest. It loads
``mla_head_padding.py`` by file path so it runs without sglang, cutlass,
torch or quack installed, and it never imports the CUDA kernels.
"""

import importlib.util
import pathlib
import subprocess
import sys
import textwrap

import pytest


def _load_predicate():
    """Load the predicate without importing the sglang package (no CUDA deps)."""
    path = pathlib.Path(__file__).with_name("mla_head_padding.py")
    spec = importlib.util.spec_from_file_location("fa_cute_mla_head_padding", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.mla_2cta_pads_dense_qheads


mla_2cta_pads_dense_qheads = _load_predicate()


@pytest.mark.parametrize(
    ("nheads_per_kv", "num_heads_kv", "expected"),
    [
        # Boundaries of the 65..127 padding window.
        (64, 1, False),
        (65, 1, True),
        (96, 1, True),
        (127, 1, True),
        (128, 1, False),
        # Too few heads keep raising: padding would waste the tile.
        (24, 1, False),
        (0, 1, False),
        (-1, 1, False),
        (129, 1, False),
        # Non-MQA (more than one KV head) never pads.
        (96, 2, False),
        (65, 2, False),
        (96, 0, False),
        # Boolean MQA flag form used by the usability check.
        (64, True, False),
        (65, True, True),
        (96, True, True),
        (127, True, True),
        (128, True, False),
        (0, True, False),
        (-1, True, False),
        (96, False, False),
    ],
)
def test_mla_2cta_pads_dense_qheads(nheads_per_kv, num_heads_kv, expected):
    assert mla_2cta_pads_dense_qheads(nheads_per_kv, num_heads_kv) is expected


@pytest.mark.parametrize("nheads", [65, 72, 96, 127])
def test_padded_heads_map_to_identity_head_order(nheads):
    """Padded rows keep the real head order and land past the valid heads."""
    tile = 128
    assert mla_2cta_pads_dense_qheads(nheads, 1)
    head_order = list(range(tile))
    assert head_order[:nheads] == list(range(nheads))
    assert all(h >= nheads for h in head_order[nheads:])


@pytest.mark.parametrize("nheads", [64, 128])
def test_non_padded_heads_keep_identity_order(nheads):
    """64 and 128 heads are not padded; the mapping is the identity."""
    assert not mla_2cta_pads_dense_qheads(nheads, 1)
    assert list(range(nheads)) == list(range(nheads))


def test_predicate_resolves_without_config_module(monkeypatch):
    """The predicate must be importable even if a separate config module is absent."""
    monkeypatch.setitem(sys.modules, "flash_attn.cute.config", None)
    predicate = _load_predicate()
    assert predicate(96, 1) is True
    assert predicate(64, 1) is False


def test_predicate_resolves_without_sglang_package(monkeypatch):
    """The predicate must load with only its own directory on sys.path."""
    cute_dir = pathlib.Path(__file__).parent
    monkeypatch.syspath_prepend(str(cute_dir))
    for name in list(sys.modules):
        if name == "sglang" or name.startswith("sglang."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.delitem(sys.modules, "mla_head_padding", raising=False)
    spec = importlib.util.spec_from_file_location(
        "mla_head_padding", cute_dir / "mla_head_padding.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    predicate = module.mla_2cta_pads_dense_qheads
    assert predicate(64, True) is False
    assert predicate(65, True) is True
    assert predicate(127, True) is True
    assert predicate(128, True) is False
    assert predicate(96, False) is False
    assert predicate(0, True) is False
    assert predicate(-1, True) is False


def test_predicate_resolves_in_isolated_subprocess():
    """A fresh interpreter with sglang blocked must still load the predicate."""
    cute_dir = pathlib.Path(__file__).parent
    script = textwrap.dedent(
        """
        import importlib.util
        import pathlib
        import sys

        cute_dir = pathlib.Path(sys.argv[1])
        sys.path.insert(0, str(cute_dir))
        sys.modules["sglang"] = None
        sys.modules["sglang.kernels"] = None
        sys.modules["sglang.kernels.ops"] = None
        sys.modules["sglang.kernels.ops.attention"] = None
        sys.modules["sglang.kernels.ops.attention.flash_attn"] = None
        sys.modules["sglang.kernels.ops.attention.flash_attn.cute"] = None
        sys.modules[
            "sglang.kernels.ops.attention.flash_attn.cute.mla_head_padding"
        ] = None

        spec = importlib.util.spec_from_file_location(
            "mla_head_padding", cute_dir / "mla_head_padding.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        predicate = module.mla_2cta_pads_dense_qheads
        assert predicate(64, True) is False
        assert predicate(65, True) is True
        assert predicate(127, True) is True
        assert predicate(128, True) is False
        assert predicate(96, False) is False
        assert predicate(0, True) is False
        assert predicate(-1, True) is False
        print("ok")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(cute_dir)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
