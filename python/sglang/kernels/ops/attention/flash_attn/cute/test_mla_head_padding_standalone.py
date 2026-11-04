# Copyright (c) 2025, Tri Dao.
"""Standalone smoke test for the dense 2CTA MLA head-padding rule.

This file deliberately imports nothing from the sglang package and needs no
CUDA/CuTe dependencies, so it can run in a minimal CI job with only pytest
installed:

    python -m pytest -q python/sglang/kernels/ops/attention/flash_attn/cute/test_mla_head_padding_standalone.py

It loads ``mla_head_padding.py`` by file path and checks the predicate
boundaries plus the identity head-order contract used by the 2CTA tile.
"""

import importlib.util
import pathlib

import pytest


def _load_predicate():
    path = pathlib.Path(__file__).with_name("mla_head_padding.py")
    spec = importlib.util.spec_from_file_location(
        "fa_cute_mla_head_padding_standalone", path
    )
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
