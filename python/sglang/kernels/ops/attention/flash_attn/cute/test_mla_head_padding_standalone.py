"""CPU-only tests for the dense 2CTA MLA Q-head padding predicate.

This module deliberately imports nothing beyond the standard library and the
sibling ``mla_head_padding.py`` (loaded by file path), so it can run on a
CPU-only runner without CUDA/CuTe, torch, quack or the sglang package. It is
loaded by ``test_pack_gqa.py`` via ``_load_standalone_tests`` and can also be
collected directly by pytest.
"""

import importlib.util
import pathlib
import sys

import pytest


def _load_predicate_module():
    path = pathlib.Path(__file__).with_name("mla_head_padding.py")
    spec = importlib.util.spec_from_file_location(
        "_fa_cute_mla_head_padding_standalone", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_module = _load_predicate_module()
mla_2cta_pads_dense_qheads = _module.mla_2cta_pads_dense_qheads
mla_2cta_accepts_dense_qheads = _module.mla_2cta_accepts_dense_qheads


@pytest.mark.parametrize(
    ("nheads_per_kv", "num_heads_kv", "expected"),
    [
        # Boundary head counts for MQA (single KV head).
        (64, 1, False),
        (65, 1, True),
        (96, 1, True),
        (127, 1, True),
        (128, 1, False),
        # Too few heads to pad: still rejected.
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


@pytest.mark.parametrize(
    ("nheads_per_kv", "num_heads_kv", "expected"),
    [
        # Heads that divide the 128-row tile are accepted as-is.
        (128, 1, True),
        (64, 1, True),
        # MQA with 65 to 127 heads is accepted by padding to the tile.
        (96, 1, True),
        (65, 1, True),
        (127, 1, True),
        # Too few heads to pad: rejected.
        (24, 1, False),
        # Non-MQA with a non-dividing ratio: rejected.
        (96, 2, False),
        # Out-of-range head counts: rejected.
        (0, 1, False),
        (-1, 1, False),
        (129, 1, False),
    ],
)
def test_mla_2cta_accepts_dense_qheads(nheads_per_kv, num_heads_kv, expected):
    assert mla_2cta_accepts_dense_qheads(nheads_per_kv, num_heads_kv) is expected


@pytest.mark.parametrize("nheads", [65, 72, 96, 127])
def test_padded_heads_keep_identity_order(nheads):
    """Padded rows keep the real head order and land past the valid heads."""
    tile = 128
    assert mla_2cta_pads_dense_qheads(nheads, 1)
    rows = list(range(tile))
    assert rows[:nheads] == list(range(nheads))
    assert all(r >= nheads for r in rows[nheads:])


@pytest.mark.parametrize("nheads", [64, 128])
def test_non_padded_heads_keep_identity_order(nheads):
    """64 and 128 heads are not padded; the mapping is the identity."""
    assert not mla_2cta_pads_dense_qheads(nheads, 1)
    assert list(range(nheads)) == list(range(nheads))


def test_predicate_resolves_without_config_module(monkeypatch):
    """The predicate must resolve even when flash_attn.cute.config is absent.

    The rule lives in this stdlib-only sibling module, so a missing package
    config module must not raise ImportError when the predicate is queried.
    """
    monkeypatch.setitem(sys.modules, "flash_attn.cute.config", None)
    module = _load_predicate_module()
    predicate = module.mla_2cta_pads_dense_qheads
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
    module = _load_predicate_module()
    predicate = module.mla_2cta_pads_dense_qheads
    assert predicate(64, True) is False
    assert predicate(65, True) is True
    assert predicate(127, True) is True
    assert predicate(128, True) is False
    assert predicate(96, False) is False
    assert predicate(0, True) is False
    assert predicate(-1, True) is False
