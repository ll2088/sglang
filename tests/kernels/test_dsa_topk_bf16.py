"""BF16 indexer logits top-k capability negotiation and fallback tests.

This file is intentionally free of torch/CUDA so it can run in a minimal
numpy+pytest environment, independent of the full sglang install.
"""

import numpy as np
import pytest

from sglang.srt.state_capturer.indexer_topk import (
    resolve_topk_logits_dtype,
    takes_bf16_logits_for,
)


# DLPack dtype codes used by the negotiation helpers.
KDL_FLOAT = 2
KDL_BFLOAT = 4


def _bf16_bits_to_float(bits: np.ndarray) -> np.ndarray:
    """Widen BF16 bit patterns to FP32 exactly (BF16 -> FP32 is lossless)."""
    return (bits.astype(np.uint32) << 16).view(np.float32)


def _bf16x2_coarse_bins(words: np.ndarray, k_bits: int = 8) -> np.ndarray:
    """Host mirror of convert_bf16x2_to_uint8x2 in kpool_topk_transform.cuh.

    Returns the coarse bins of the two BF16 scores packed in each 32-bit word,
    low half first, as an (n, 2) uint8 array.
    """
    words = words.astype(np.uint32)
    sign = words & np.uint32(0x80008000)
    # Per half: ~bits if negative, else bits | 0x8000.
    key = words ^ ((sign - (sign >> np.uint32(15))) | np.uint32(0x80008000))
    lo = (key & np.uint32(0xFFFF)) >> np.uint32(16 - k_bits)
    hi = (key >> np.uint32(32 - k_bits)) & np.uint32(0xFF)
    return np.stack([lo.astype(np.uint8), hi.astype(np.uint8)], axis=1)


def _scalar_coarse_bin(bits: int, k_bits: int = 8) -> int:
    """Host mirror of convert_bf16_to_uint8 for a single BF16 bit pattern."""
    bits = int(bits) & 0xFFFF
    key = (~bits) & 0xFFFF if (bits & 0x8000) else (bits | 0x8000)
    return (key >> (16 - k_bits)) & 0xFF


@pytest.mark.parametrize("row_stride", [8, 16, 1024, 65536])
@pytest.mark.parametrize("backend", ["cooperative", "persistent"])
def test_bf16_native_read_when_aligned(row_stride: int, backend: str) -> None:
    assert takes_bf16_logits_for(KDL_BFLOAT, 16, 1, row_stride)
    assert resolve_topk_logits_dtype(KDL_BFLOAT, 16, 1, row_stride, backend) == "bf16"


@pytest.mark.parametrize("row_stride", [1, 4, 7, 15, 1023])
@pytest.mark.parametrize("backend", ["cooperative", "persistent"])
def test_bf16_falls_back_when_unaligned(row_stride: int, backend: str) -> None:
    assert not takes_bf16_logits_for(KDL_BFLOAT, 16, 1, row_stride)
    assert resolve_topk_logits_dtype(KDL_BFLOAT, 16, 1, row_stride, backend) == "fp32"


@pytest.mark.parametrize(
    "dtype_code,dtype_bits,dtype_lanes",
    [
        (KDL_FLOAT, 32, 1),
        (KDL_BFLOAT, 32, 1),
        (KDL_BFLOAT, 16, 2),
        (KDL_FLOAT, 16, 1),
    ],
)
def test_non_bf16_dtypes_fall_back(dtype_code: int, dtype_bits: int, dtype_lanes: int) -> None:
    assert not takes_bf16_logits_for(dtype_code, dtype_bits, dtype_lanes, 1024)
    assert resolve_topk_logits_dtype(dtype_code, dtype_bits, dtype_lanes, 1024) == "fp32"


def test_unknown_backend_falls_back() -> None:
    assert resolve_topk_logits_dtype(KDL_BFLOAT, 16, 1, 1024, "deep_select") == "fp32"


def test_bf16x2_packed_bins_match_scalar() -> None:
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 1 << 16, size=4096, dtype=np.uint16)
    words = (bits[0::2].astype(np.uint32) | (bits[1::2].astype(np.uint32) << 16)).astype(np.uint32)
    packed = _bf16x2_coarse_bins(words)
    expected = np.array(
        [[_scalar_coarse_bin(bits[2 * i]), _scalar_coarse_bin(bits[2 * i + 1])] for i in range(len(words))],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(packed, expected)


def test_bf16x2_packed_bins_handle_signed_zero_and_inf() -> None:
    # +0, -0, +inf, -inf as BF16 bit patterns.
    bits = np.array([0x0000, 0x8000, 0x7F80, 0xFF80], dtype=np.uint16)
    words = np.array(
        [int(bits[0]) | (int(bits[1]) << 16), int(bits[2]) | (int(bits[3]) << 16)],
        dtype=np.uint32,
    )
    packed = _bf16x2_coarse_bins(words)
    expected = np.array(
        [[_scalar_coarse_bin(bits[0]), _scalar_coarse_bin(bits[1])],
         [_scalar_coarse_bin(bits[2]), _scalar_coarse_bin(bits[3])]],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(packed, expected)


def test_bf16_widening_is_exact() -> None:
    rng = np.random.default_rng(1)
    bits = rng.integers(0, 1 << 16, size=1024, dtype=np.uint16)
    widened = _bf16_bits_to_float(bits)
    # Widening BF16 to FP32 is exact: the low 16 mantissa bits are zero.
    assert np.all((widened.view(np.uint32) & np.uint32(0xFFFF)) == 0)


def _topk_indices(values: np.ndarray, k: int) -> np.ndarray:
    """Deterministic top-k by value, ties broken by lower index."""
    order = np.lexsort((np.arange(values.size), -values))
    return order[:k]


@pytest.mark.parametrize("length", [300, 5000, 16384, 40000])
@pytest.mark.parametrize("k", [512, 2048])
def test_bf16_and_fp32_select_same_values(length: int, k: int) -> None:
    rng = np.random.default_rng(length + k)
    fp32 = rng.standard_normal(length).astype(np.float32)
    # Round-trip through BF16 so both paths see the same representable values.
    bf16_bits = (fp32.view(np.uint32) >> np.uint32(16)).astype(np.uint16)
    bf16 = _bf16_bits_to_float(bf16_bits)

    kk = min(k, length)
    idx_fp32 = _topk_indices(fp32, kk)
    idx_bf16 = _topk_indices(bf16, kk)
    np.testing.assert_array_equal(
        np.sort(fp32[idx_fp32]),
        np.sort(bf16[idx_bf16]),
    )


def test_bf16_unaligned_row_falls_back_to_fp32() -> None:
    # A row stride that is not a multiple of 8 must take the FP32 path.
    assert resolve_topk_logits_dtype(KDL_BFLOAT, 16, 1, 1023, "cooperative") == "fp32"
    assert resolve_topk_logits_dtype(KDL_BFLOAT, 16, 1, 1023, "persistent") == "fp32"
