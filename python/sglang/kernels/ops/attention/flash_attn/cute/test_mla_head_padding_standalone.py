# Copyright (c) 2025, Tri Dao.
"""Stdlib-only tests for the dense 2CTA MLA Q-head padding rule.

These tests deliberately avoid importing sglang, torch, cutlass or quack so a
minimal CI job can run them without installing the full dependency tree. The
rule under test lives in ``mla_head_padding.py`` and is loaded by file path.
"""

import importlib.util
import pathlib
import sys

import pytest


cpu_only = pytest.mark.cpu_only


def _load_module():
    path = pathlib.Path(__file__).with_name("mla_head_padding.py")
    spec = importlib.util.spec_from_file_location(
        "_fa_cute_mla_head_padding_standalone", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mla_2cta_pads_dense_qheads(nheads_per_kv, num_heads_kv):
    return _load_module().mla_2cta_pads_dense_qheads(nheads_per_kv, num_heads_kv)


def mla_2cta_accepts_dense_qheads(nheads_per_kv, num_heads_kv):
    return _load_module().mla_2cta_accepts_dense_qheads(nheads_per_kv, num_heads_kv)


@cpu_only
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


@cpu_only
@pytest.mark.parametrize(
    ("nheads_per_kv", "num_heads_kv", "expected"),
    [
        (128, 1, True),
        (64, 1, True),
        (96, 1, True),
        (65, 1, True),
        (127, 1, True),
        (24, 1, False),
        (96, 2, False),
        (0, 1, False),
        (-1, 1, False),
    ],
)
def test_mla_2cta_accepts_dense_qheads(nheads_per_kv, num_heads_kv, expected):
    assert mla_2cta_accepts_dense_qheads(nheads_per_kv, num_heads_kv) is expected


@cpu_only
@pytest.mark.parametrize("nheads", [65, 72, 96, 127])
def test_padded_heads_keep_identity_order(nheads):
    """Padded rows keep the real head order and land past the valid heads."""
    tile = 128
    assert mla_2cta_pads_dense_qheads(nheads, 1)
    rows = list(range(tile))
    assert rows[:nheads] == list(range(nheads))
    assert all(r >= nheads for r in rows[nheads:])


@cpu_only
@pytest.mark.parametrize("nheads", [64, 128])
def test_non_padded_heads_keep_identity_order(nheads):
    """64 and 128 heads are not padded; the mapping is the identity."""
    assert not mla_2cta_pads_dense_qheads(nheads, 1)
    assert list(range(nheads)) == list(range(nheads))


@cpu_only
def test_predicate_loads_without_config_module(monkeypatch):
    """The predicate must resolve even if flash_attn.cute.config is absent."""
    monkeypatch.setitem(sys.modules, "flash_attn.cute.config", None)
    monkeypatch.setitem(sys.modules, "flash_attn", None)
    monkeypatch.setitem(sys.modules, "flash_attn.cute", None)
    predicate = mla_2cta_pads_dense_qheads
    assert predicate(65, 1) is True
    assert predicate(127, 1) is True
    assert predicate(128, 1) is False
    assert predicate(96, 2) is False


@cpu_only
def test_predicate_loads_without_sglang_or_cuda_deps(monkeypatch):
    """The predicate must load with sglang/cutlass/torch/quack blocked."""
    for name in list(sys.modules):
        if name == "sglang" or name.startswith("sglang."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "sglang", None)
    monkeypatch.setitem(sys.modules, "cutlass", None)
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "quack", None)
    predicate = mla_2cta_pads_dense_qheads
    assert predicate(64, True) is False
    assert predicate(65, True) is True
    assert predicate(127, True) is True
    assert predicate(128, True) is False
    assert predicate(96, False) is False
    assert predicate(0, True) is False
    assert predicate(-1, True) is False
