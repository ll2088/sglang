# Copyright (c) 2026, SGLang Team.
"""Torch-free unit tests for the FA4 SM90 d512 FP8-MMA routing policy.

Only pytest is required, so this file can run in a standalone job before the
full sglang dependency set is installed.
"""

import pytest

from sglang.kernels.ops.attention.fa4_sm90_fp8_policy import (
    SM90_FP8_MMA_HEAD_DIM,
    SM90_FP8_MMA_MAX_SPLITS,
    is_sm90_fp8_kv_mma_dtypes,
    normalize_kv_descale,
    select_num_splits,
)


def test_head_dim_constant():
    assert SM90_FP8_MMA_HEAD_DIM == 512


def test_select_num_splits_fa3_capped_at_32():
    assert (
        select_num_splits(
            fa_version=3,
            max_seqlen_k=8192,
            num_heads=32,
            num_kv_heads=8,
            num_sms=132,
            is_fp8_mma=False,
        )
        == 32
    )


def test_select_num_splits_non_fp8_uses_kernel_default():
    assert (
        select_num_splits(
            fa_version=4,
            max_seqlen_k=8192,
            num_heads=32,
            num_kv_heads=8,
            num_sms=132,
            is_fp8_mma=False,
        )
        == 0
    )


@pytest.mark.parametrize(
    "max_seqlen_k,expected",
    [
        (0, 1),
        (64, 1),
        (256, 1),
        (1024, 4),
        (8192, SM90_FP8_MMA_MAX_SPLITS),
    ],
)
def test_select_num_splits_fp8_monotonic(max_seqlen_k, expected):
    got = select_num_splits(
        fa_version=4,
        max_seqlen_k=max_seqlen_k,
        num_heads=32,
        num_kv_heads=8,
        num_sms=132,
        is_fp8_mma=True,
    )
    assert got == expected
    assert 1 <= got <= SM90_FP8_MMA_MAX_SPLITS


def test_select_num_splits_fp8_bounded_by_sms():
    got = select_num_splits(
        fa_version=4,
        max_seqlen_k=8192,
        num_heads=32,
        num_kv_heads=8,
        num_sms=4,
        is_fp8_mma=True,
    )
    assert got == 1


def test_select_num_splits_fp8_invalid_inputs():
    assert (
        select_num_splits(
            fa_version=4,
            max_seqlen_k=8192,
            num_heads=0,
            num_kv_heads=8,
            num_sms=132,
            is_fp8_mma=True,
        )
        == 1
    )


def test_normalize_kv_descale_none_and_scalar():
    assert normalize_kv_descale(None, batch=2, heads=4) is None
    assert normalize_kv_descale(0.5, batch=2, heads=4) == 0.5


def test_normalize_kv_descale_rejects_unsupported_type():
    with pytest.raises(ValueError):
        normalize_kv_descale([1.0, 2.0], batch=2, heads=2)


def test_is_sm90_fp8_kv_mma_dtypes_rejects_non_sm90():
    assert not is_sm90_fp8_kv_mma_dtypes(
        None,
        None,
        None,
        512,
        512,
        512,
        True,
        (None, None),
        device_capability_major=10,
    )


def test_is_sm90_fp8_kv_mma_dtypes_requires_torch():
    torch = pytest.importorskip("torch")
    assert is_sm90_fp8_kv_mma_dtypes(
        torch.bfloat16,
        torch.float8_e4m3fn,
        torch.float8_e4m3fn,
        512,
        512,
        512,
        True,
        (None, None),
        device_capability_major=9,
    )
    assert not is_sm90_fp8_kv_mma_dtypes(
        torch.bfloat16,
        torch.float8_e4m3fn,
        torch.float8_e4m3fn,
        256,
        256,
        256,
        True,
        (None, None),
        device_capability_major=9,
    )


def test_is_sm90_fp8_kv_mma_dtypes_rejects_noncausal_full():
    torch = pytest.importorskip("torch")
    with pytest.raises(NotImplementedError):
        is_sm90_fp8_kv_mma_dtypes(
            torch.bfloat16,
            torch.float8_e4m3fn,
            torch.float8_e4m3fn,
            512,
            512,
            512,
            False,
            (None, None),
            device_capability_major=9,
        )
