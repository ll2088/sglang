# Copyright (c) 2025, Tri Dao.
"""CPU-only tests for the dense 2CTA MLA Q-head padding rule.

These tests do not import CUDA/CuTe kernels: they exercise the pure-Python
predicate, the pack_gqa head-order mapping, and the config-module fallback so
they can run on any runner.
"""

import importlib.util
import pathlib
import sys

import pytest


def _load_predicate():
    """Load the predicate without importing the sglang package (no CUDA deps)."""
    path = pathlib.Path(__file__).with_name("pack_gqa.py")
    spec = importlib.util.spec_from_file_location("fa_cute_pack_gqa", path)
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
    """Padded rows must keep the real head order and land past the valid heads.

    The kernel packs a token's Q heads into the 128-row tile; the first
    ``nheads`` rows are the real heads in order and the remaining rows are
    padding that must not alias a real head.
    """
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
    """The predicate must be importable even if a separate config module is absent.

    The rule lives in pack_gqa, so importing it must not require
    ``flash_attn.cute.config`` to exist.
    """
    monkeypatch.setitem(sys.modules, "flash_attn.cute.config", None)
    predicate = _load_predicate()
    assert predicate(96, 1) is True
    assert predicate(64, 1) is False
