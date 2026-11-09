# Copyright (c) 2026, SGLang Team.
"""Torch-free tests for the FA4 SM90 d512 FP8-MMA routing policy.

Runs on a CPU-only runner with only pytest installed, so CI can validate the
policy module even when the sglang runtime dependencies are unavailable.
"""

import importlib.util
import pathlib
import sys

import pytest

_POLICY_PATH = pathlib.Path(__file__).with_name("fa4_sm90_fp8_policy.py")
_MODULE_NAME = "_fa4_sm90_fp8_policy_standalone_under_test"
_spec = importlib.util.spec_from_file_location(_MODULE_NAME, _POLICY_PATH)
fa4 = importlib.util.module_from_spec(_spec)
sys.modules[_MODULE_NAME] = fa4
_spec.loader.exec_module(fa4)


def test_module_imports_without_torch():
    assert fa4.SM90_FP8_MMA_HEAD_DIM == 512
    assert fa4.SM90_FP8_MMA_MAX_SPLITS == 4


def test_scalar_descale_is_returned_as_float():
    assert fa4.normalize_kv_descale(0.5, batch=2, heads=4) == 0.5
    assert fa4.normalize_kv_descale(None, batch=2, heads=4) is None


def test_split_selection_fa3_cap_is_32():
    assert (
        fa4.select_num_splits(
            fa_version=3,
            max_seqlen_k=16384,
            num_heads=8,
            num_kv_heads=1,
            num_sms=132,
            is_fp8_mma=True,
        )
        == 32
    )


def test_split_selection_non_fp8_returns_zero():
    assert (
        fa4.select_num_splits(
            fa_version=4,
            max_seqlen_k=16384,
            num_heads=8,
            num_kv_heads=1,
            num_sms=132,
            is_fp8_mma=False,
        )
        == 0
    )


def test_split_selection_bounds_and_monotonicity():
    short = fa4.select_num_splits(
        fa_version=4, max_seqlen_k=256, num_heads=8, num_kv_heads=1,
        num_sms=132, is_fp8_mma=True,
    )
    long = fa4.select_num_splits(
        fa_version=4, max_seqlen_k=16384, num_heads=8, num_kv_heads=1,
        num_sms=132, is_fp8_mma=True,
    )
    assert 1 <= short <= long <= fa4.SM90_FP8_MMA_MAX_SPLITS
    assert long <= 16384 // fa4.SM90_FP8_MMA_TILE_N


def test_split_selection_degenerate_inputs():
    assert fa4.select_num_splits(
        fa_version=4, max_seqlen_k=0, num_heads=8, num_kv_heads=1,
        num_sms=132, is_fp8_mma=True,
    ) == 1
    assert fa4.select_num_splits(
        fa_version=4, max_seqlen_k=4096, num_heads=8, num_kv_heads=1,
        num_sms=0, is_fp8_mma=True,
    ) == 1


def test_device_capability_helper_accepts_explicit_major():
    assert fa4._device_capability_major(9) == 9
    assert fa4._device_capability_major(10) == 10


def test_routing_gate_returns_false_without_torch():
    # Without torch the gate must fail closed instead of raising.
    assert fa4.is_sm90_fp8_kv_mma(None, None, None, True, (None, None)) is False


def test_descale_rejects_unsupported_type():
    with pytest.raises(ValueError):
        fa4.normalize_kv_descale(object(), batch=2, heads=4)
