# Copyright (c) 2026, SGLang Team.
"""Standard-library-only checks for the FA4 SM90 FP8 routing policy.

Runs without torch so CI can validate the pure host logic even when the
CUDA/torch environment is unavailable.
"""

import importlib.util
import pathlib
import sys

import pytest

_POLICY_PATH = pathlib.Path(__file__).with_name("fa4_sm90_fp8_policy.py")
_NAME = "_fa4_sm90_fp8_policy_standalone_under_test"
_spec = importlib.util.spec_from_file_location(_NAME, _POLICY_PATH)
fa4 = importlib.util.module_from_spec(_spec)
sys.modules[_NAME] = fa4
_spec.loader.exec_module(fa4)


def test_module_imports_without_torch():
    assert fa4.SM90_FP8_MMA_HEAD_DIM == 512
    assert fa4.SM90_FP8_MMA_MAX_SPLITS == 4


def test_scalar_descale_is_torch_free():
    assert fa4.normalize_kv_descale(None, batch=2, heads=4) is None
    assert fa4.normalize_kv_descale(0.5, batch=2, heads=4) == 0.5


def test_split_selection_bounds():
    assert fa4.select_num_splits(
        fa_version=3, max_seqlen_k=16384, num_heads=8, num_kv_heads=1,
        num_sms=132, is_fp8_mma=True,
    ) == 32
    assert fa4.select_num_splits(
        fa_version=4, max_seqlen_k=16384, num_heads=8, num_kv_heads=1,
        num_sms=132, is_fp8_mma=False,
    ) == 0
    splits = fa4.select_num_splits(
        fa_version=4, max_seqlen_k=4096, num_heads=8, num_kv_heads=1,
        num_sms=132, is_fp8_mma=True,
    )
    assert 1 <= splits <= fa4.SM90_FP8_MMA_MAX_SPLITS


def test_split_selection_degenerate_inputs():
    assert fa4.select_num_splits(
        fa_version=4, max_seqlen_k=0, num_heads=8, num_kv_heads=1,
        num_sms=132, is_fp8_mma=True,
    ) == 1
    assert fa4.select_num_splits(
        fa_version=4, max_seqlen_k=4096, num_heads=8, num_kv_heads=1,
        num_sms=0, is_fp8_mma=True,
    ) == 1


def test_routing_returns_false_without_torch():
    class _T:
        dtype = object()
        shape = (1, 8, 512)

    assert fa4.is_sm90_fp8_kv_mma(_T(), _T(), _T(), True, (None, None)) is False
