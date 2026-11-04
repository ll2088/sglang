# Copyright (c) 2025, Tri Dao.
"""CPU-only tests for the dense 2CTA MLA Q-head padding rule.

This file deliberately imports nothing from the sglang package and needs no
CUDA/CuTe, torch, or quack dependencies, so it can run in a minimal CI job
with only pytest installed:

    python -m pytest -q python/sglang/kernels/ops/attention/flash_attn/cute/test_mla_head_padding_cpu.py

It loads ``mla_head_padding.py`` by file path and checks the predicate
boundaries plus the identity head-order contract used by the 2CTA tile.
"""

import importlib.util
import pathlib
import sys

import pytest


def _load_predicate():
    path = pathlib.Path(__file__).with_name("mla_head_padding.py")
    spec = importlib.util.spec_from_file_location(
        "fa_cute_mla_head_padding_cpu", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.mla_2cta_pads_dense_qheads


def test_predicate_does_not_import_sglang():
    """Loading the predicate must not pull in the sglang package."""
    for name in list(sys.modules):
        if name == "sglang" or name.startswith("sglang."):
            del sys.modules[name]

    _load_predicate()
    assert "sglang" not in sys.modules


def test_predicate_does_not_import_cutlass():
    """The predicate must stay importable without CUDA/CuTe dependencies."""
    for name in list(sys.modules):
        if name == "cutlass" or name.startswith("cutlass."):
            del sys.modules[name]

    _load_predicate()
    assert "cutlass" not in sys.modules


def test_predicate_does_not_import_third_party():
    """The predicate must not require torch, quack, or other third-party deps."""
    for name in list(sys.modules):
        if name in ("torch", "quack") or name.startswith(("torch.", "quack.")):
            del sys.modules[name]

    _load_predicate()
    assert "torch" not in sys.modules
    assert "quack" not in sys.modules


def test_predicate_resolves_without_config_module(monkeypatch):
    """The predicate must load even when a separate config module is absent."""
    monkeypatch.setitem(sys.modules, "flash_attn.cute.config", None)
    predicate = _load_predicate()
    assert predicate(96, 1) is True
    assert predicate(64, 1) is False


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
def test_predicate_boundaries(nheads_per_kv, num_heads_kv, expected):
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
