# Copyright (c) 2025, Tri Dao.
"""CPU-only tests for the dense 2CTA MLA Q-head padding rule.

This module imports nothing beyond the standard library and the sibling
``mla_head_padding.py`` (loaded by file path), so it runs on a minimal CI job
without sglang, cutlass, quack or torch installed.
"""

import importlib.util
import pathlib

import pytest

cpu_only = pytest.mark.cpu_only


def _load_rule_module():
    path = pathlib.Path(__file__).with_name("mla_head_padding.py")
    spec = importlib.util.spec_from_file_location(
        "_fa_cute_mla_head_padding_cpu", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@cpu_only
@pytest.mark.parametrize(
    ("nheads_per_kv", "num_heads_kv", "expected"),
    [
        (0, 1, False),
        (-1, 1, False),
        (1, 1, False),
        (24, 1, False),
        (64, 1, False),
        (65, 1, True),
        (72, 1, True),
        (96, 1, True),
        (127, 1, True),
        (128, 1, False),
        (129, 1, False),
        (96, 2, False),
        (96, 0, False),
        (96, 4, False),
    ],
)
def test_mla_2cta_pads_dense_qheads_boundaries(
    nheads_per_kv, num_heads_kv, expected
):
    rule = _load_rule_module()
    assert rule.mla_2cta_pads_dense_qheads(nheads_per_kv, num_heads_kv) is expected


@cpu_only
@pytest.mark.parametrize(
    ("nheads_per_kv", "is_mqa", "expected"),
    [
        (0, True, False),
        (-1, True, False),
        (64, True, False),
        (65, True, True),
        (96, True, True),
        (127, True, True),
        (128, True, False),
        (96, False, False),
        (65, False, False),
    ],
)
def test_mla_2cta_pads_dense_qheads_boolean_mqa_flag(
    nheads_per_kv, is_mqa, expected
):
    rule = _load_rule_module()
    assert rule.mla_2cta_pads_dense_qheads(nheads_per_kv, is_mqa) is expected


@cpu_only
@pytest.mark.parametrize(
    ("nheads_per_kv", "num_heads_kv", "expected"),
    [
        (0, 1, False),
        (-1, 1, False),
        (1, 1, True),
        (2, 1, True),
        (24, 1, False),
        (64, 1, True),
        (65, 1, True),
        (96, 1, True),
        (127, 1, True),
        (128, 1, True),
        (129, 1, False),
        (96, 2, False),
        (128, 2, True),
        (64, 2, True),
    ],
)
def test_mla_2cta_accepts_dense_qheads_boundaries(
    nheads_per_kv, num_heads_kv, expected
):
    rule = _load_rule_module()
    assert rule.mla_2cta_accepts_dense_qheads(nheads_per_kv, num_heads_kv) is expected


@cpu_only
def test_accepts_rule_agrees_with_padding_rule():
    rule = _load_rule_module()
    for nheads_per_kv in range(1, 129):
        for num_heads_kv in (1, 2, 4):
            divides = 128 % nheads_per_kv == 0
            pads = rule.mla_2cta_pads_dense_qheads(nheads_per_kv, num_heads_kv)
            accepts = rule.mla_2cta_accepts_dense_qheads(nheads_per_kv, num_heads_kv)
            assert accepts is (divides or pads)


@cpu_only
def test_rule_module_has_no_heavy_imports():
    """The rule module must stay importable without CUDA/CuTe/sglang."""
    source = pathlib.Path(__file__).with_name("mla_head_padding.py").read_text()
    for forbidden in ("import cutlass", "import torch", "import quack", "import sglang"):
        assert forbidden not in source
