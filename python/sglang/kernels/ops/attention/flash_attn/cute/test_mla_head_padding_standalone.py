# Copyright (c) 2025, Tri Dao.
"""CPU-only tests for the dense 2CTA MLA Q-head padding rule.

These tests import only the pure-Python predicate module by file path, so they
run without CUDA/CuTe, torch, quack, or the sglang package. They are the
minimal-dependency CI entry point for the padding rule.
"""

import importlib.util
import pathlib
import subprocess
import sys
import textwrap

import pytest


def _load_predicate():
    path = pathlib.Path(__file__).with_name("mla_head_padding.py")
    spec = importlib.util.spec_from_file_location("fa_cute_mla_head_padding", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.mla_2cta_pads_dense_qheads


mla_2cta_pads_dense_qheads = _load_predicate()


@pytest.mark.parametrize(
    ("nheads_per_kv", "num_heads_kv", "expected"),
    [
        (64, 1, False),
        (65, 1, True),
        (96, 1, True),
        (127, 1, True),
        (128, 1, False),
        (24, 1, False),
        (0, 1, False),
        (-1, 1, False),
        (129, 1, False),
        (96, 2, False),
        (65, 2, False),
        (96, 0, False),
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
def test_padded_heads_keep_identity_order(nheads):
    tile = 128
    assert mla_2cta_pads_dense_qheads(nheads, 1)
    rows = list(range(tile))
    assert rows[:nheads] == list(range(nheads))
    assert all(r >= nheads for r in rows[nheads:])


@pytest.mark.parametrize("nheads", [64, 128])
def test_non_padded_heads_keep_identity_order(nheads):
    assert not mla_2cta_pads_dense_qheads(nheads, 1)
    assert list(range(nheads)) == list(range(nheads))


def test_predicate_resolves_without_config_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "flash_attn.cute.config", None)
    predicate = _load_predicate()
    assert predicate(96, 1) is True
    assert predicate(64, 1) is False


def test_predicate_resolves_without_sglang_or_cuda_deps(monkeypatch):
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


def test_predicate_resolves_in_isolated_subprocess():
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
