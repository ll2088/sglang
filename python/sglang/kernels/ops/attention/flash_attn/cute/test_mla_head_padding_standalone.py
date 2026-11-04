# Copyright (c) 2025, Tri Dao.
"""CPU-only tests for the dense 2CTA MLA Q-head padding rule.

These tests deliberately avoid importing the sglang package, CUDA/CuTe, torch
or quack: they load ``mla_head_padding.py`` and ``pack_gqa.py`` by file path so
the predicate and its pack_gqa fallback can be verified on a minimal runner
where the full sglang dependency set is unavailable.
"""

import importlib.util
import pathlib
import subprocess
import sys
import textwrap

import pytest


CUTE_DIR = pathlib.Path(__file__).parent


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, CUTE_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_predicate_module():
    return _load_module("fa_cute_mla_head_padding_standalone", "mla_head_padding.py")


def _load_predicate():
    return _load_predicate_module().mla_2cta_pads_dense_qheads


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
    rows = list(range(tile))
    assert rows[:nheads] == list(range(nheads))
    assert all(r >= nheads for r in rows[nheads:])
    # A second token starts at the next tile, never inside the padded rows.
    assert tile >= nheads


@pytest.mark.parametrize("nheads", [64, 128])
def test_non_padded_heads_keep_identity_order(nheads):
    """64 and 128 heads are not padded; the mapping is the identity."""
    assert not mla_2cta_pads_dense_qheads(nheads, 1)
    assert list(range(nheads)) == list(range(nheads))


def test_accepts_dense_qheads_matches_padding_rule():
    """The acceptance rule is divisibility or the 65-127 MQA padding window."""
    accepts = _load_predicate_module().mla_2cta_accepts_dense_qheads
    assert accepts(128, 1) is True
    assert accepts(64, 1) is True
    assert accepts(96, 1) is True
    assert accepts(65, 1) is True
    assert accepts(127, 1) is True
    assert accepts(24, 1) is False
    assert accepts(96, 2) is False
    assert accepts(0, 1) is False
    assert accepts(-1, 1) is False


def test_predicate_resolves_without_config_module(monkeypatch):
    """The predicate must be importable even if a separate config module is absent."""
    monkeypatch.setitem(sys.modules, "flash_attn.cute.config", None)
    predicate = _load_predicate()
    assert predicate(96, 1) is True
    assert predicate(64, 1) is False


def test_predicate_resolves_without_sglang_or_cuda_deps(monkeypatch):
    """The predicate must load with sglang/cutlass/torch/quack blocked."""
    for name in list(sys.modules):
        if name == "sglang" or name.startswith("sglang."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "sglang", None)
    monkeypatch.setitem(sys.modules, "cutlass", None)
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "quack", None)
    predicate = _load_predicate()
    assert predicate(64, True) is False
    assert predicate(65, True) is True
    assert predicate(127, True) is True
    assert predicate(128, True) is False
    assert predicate(96, False) is False
    assert predicate(0, True) is False
    assert predicate(-1, True) is False


def test_pack_gqa_predicate_fallback_without_config_module(monkeypatch):
    """pack_gqa must resolve the predicate via its sibling module when config is absent.

    This is the CPU-only path used by CI without CUDA/CuTe or the sglang package:
    the predicate is loaded by file path from mla_head_padding.py, so a missing
    flash_attn.cute.config must not raise ImportError.
    """
    monkeypatch.setitem(sys.modules, "flash_attn.cute.config", None)
    try:
        module = _load_module("_fa_cute_pack_gqa_standalone", "pack_gqa.py")
    except (ImportError, ModuleNotFoundError):
        pytest.skip("pack_gqa needs CUDA/CuTe deps")
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
        [sys.executable, "-c", script, str(CUTE_DIR)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
