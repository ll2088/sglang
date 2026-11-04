# Copyright (c) 2025, Tri Dao.
"""CPU-only pytest coverage for the dense 2CTA MLA Q-head padding rule.

These tests import nothing outside the Python standard library so they run in a
minimal CI job without sglang's full dependency tree. They load
``mla_head_padding.py`` by file path, mirroring pack_gqa's sibling-module
fallback, and cover the 64/65/127/128 boundaries, the boolean MQA flag form,
the config-missing scenario, and the pack_gqa head-order mapping.
"""

import importlib.util
import pathlib
import sys

import pytest


_HERE = pathlib.Path(__file__).resolve().parent

cpu_only = pytest.mark.cpu_only


class _Missing:
    pass


_MISSING = _Missing()


def _load_by_path(name, filename):
    path = _HERE / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_predicate():
    return _load_by_path("_fa_cute_mla_head_padding_cpu_test", "mla_head_padding.py")


def _load_pack_gqa_predicate_only():
    return _load_predicate().mla_2cta_pads_dense_qheads


@cpu_only
@pytest.mark.parametrize(
    ("heads", "heads_kv", "expected"),
    [
        (64, 1, False),
        (65, 1, True),
        (96, 1, True),
        (127, 1, True),
        (128, 1, False),
        (96, 2, False),
        (0, 1, False),
        (-1, 1, False),
    ],
)
def test_predicate_boundaries(heads, heads_kv, expected):
    predicate = _load_predicate().mla_2cta_pads_dense_qheads
    assert predicate(heads, heads_kv) is expected


@cpu_only
@pytest.mark.parametrize(
    ("heads", "is_mqa", "expected"),
    [
        (64, True, False),
        (65, True, True),
        (127, True, True),
        (128, True, False),
        (96, False, False),
        (0, True, False),
        (-1, True, False),
    ],
)
def test_predicate_accepts_boolean_mqa_flag(heads, is_mqa, expected):
    predicate = _load_predicate().mla_2cta_pads_dense_qheads
    assert predicate(heads, is_mqa) is expected


@cpu_only
@pytest.mark.parametrize(
    ("heads", "heads_kv", "expected"),
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
def test_accepts_rule(heads, heads_kv, expected):
    accepts = _load_predicate().mla_2cta_accepts_dense_qheads
    assert accepts(heads, heads_kv) is expected


@cpu_only
@pytest.mark.parametrize("heads", [1, 2, 4, 8, 16, 32, 64, 128])
def test_non_padding_paths_unchanged(heads):
    """Head counts that divide 128 must not take the padding path."""
    predicate = _load_predicate().mla_2cta_pads_dense_qheads
    assert predicate(heads, 1) is False


@cpu_only
def test_predicate_resolves_without_config_module(monkeypatch):
    """The predicate must resolve when flash_attn.cute.config is absent."""
    for name in (
        "flash_attn",
        "flash_attn.cute",
        "flash_attn.cute.config",
        "sglang",
        "sglang.kernels",
        "sglang.kernels.ops",
        "sglang.kernels.ops.attention",
        "sglang.kernels.ops.attention.flash_attn",
        "sglang.kernels.ops.attention.flash_attn.cute",
        "sglang.kernels.ops.attention.flash_attn.cute.mla_head_padding",
    ):
        monkeypatch.setitem(sys.modules, name, None)
    predicate = _load_pack_gqa_predicate_only()
    assert predicate(64, True) is False
    assert predicate(65, True) is True
    assert predicate(127, True) is True
    assert predicate(128, True) is False
    assert predicate(96, False) is False
    assert predicate(0, True) is False
    assert predicate(-1, True) is False


@cpu_only
@pytest.mark.parametrize("qhead_per_kvhead", [65, 72, 96, 127])
def test_pack_gqa_head_order_mapping(qhead_per_kvhead):
    """Padded rows must not alias a real head of the next token.

    Mirrors ``PackGQA.compute_ptr`` index arithmetic: ``m_idx = idx //
    qhead_per_kvhead``, ``h_idx = idx - m_idx * qhead_per_kvhead``.
    """
    tile = 128
    seqlen = 3
    for block in range(0, seqlen * tile, tile):
        for row in range(tile):
            idx = block + row
            m_idx = idx // qhead_per_kvhead
            h_idx = idx - m_idx * qhead_per_kvhead
            if row < qhead_per_kvhead:
                assert m_idx < seqlen
                assert 0 <= h_idx < qhead_per_kvhead
                assert idx == m_idx * qhead_per_kvhead + h_idx
            else:
                assert h_idx >= qhead_per_kvhead or m_idx >= seqlen
