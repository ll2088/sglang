"""Standalone tests for the DSA indexer top-k BF16 path.

These tests deliberately avoid importing torch or sglang so that they can be
collected and run in a minimal CI environment.  They mirror the host-side
predicates and the packed BF16x2 histogram extraction used by
``python/sglang/kernels/jit/csrc/dsa/kpool_topk_transform.cuh`` and
``python/sglang/kernels/jit/csrc/dsa_gfx950/topk_transform.cuh``.
"""

import struct

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Host-side mirrors of the device helpers.
# ---------------------------------------------------------------------------


def fp32_to_bf16_bits(x: float) -> int:
    """Round-to-nearest-even FP32 -> BF16, matching __float2bfloat16_rn."""
    bits = struct.unpack("<I", struct.pack("<f", float(x)))[0]
    # NaN stays NaN; infinities keep their exponent.
    lsb = (bits >> 16) & 1
    rounding = 0x7FFF + lsb
    bits = (bits + rounding) & 0xFFFFFFFF
    return (bits >> 16) & 0xFFFF


def bf16_bits_to_float(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", (bits & 0xFFFF) << 16))[0]


def convert_bf16_to_uint8(bits: int) -> int:
    """Ordered 16-bit key of a BF16 score, top 8 bits (kpool path)."""
    key = (~bits) & 0xFFFF if (bits & 0x8000) else (bits | 0x8000)
    return (key >> 8) & 0xFF


def convert_bf16x2_to_uint8x2(w: int):
    """Packed form of convert_bf16_to_uint8 for two BF16 scores in `w`."""
    lo = w & 0xFFFF
    hi = (w >> 16) & 0xFFFF
    return convert_bf16_to_uint8(lo), convert_bf16_to_uint8(hi)


def convert_to_uint8_fp32(x: float) -> int:
    """FP32 -> FP16 ordered key, top 8 bits (kpool FP32 path)."""
    h = np.float16(x)
    bits = struct.unpack("<H", struct.pack("<e", float(h)))[0]
    key = (~bits) & 0xFFFF if (bits & 0x8000) else (bits | 0x8000)
    return (key >> 8) & 0xFF


def order_key16_fp32(x: float) -> int:
    """gfx950 order_key16: FP32 -> FP16 ordered 16-bit key."""
    h = np.float16(x)
    bits = struct.unpack("<H", struct.pack("<e", float(h)))[0]
    return (~bits) & 0xFFFF if (bits & 0x8000) else (bits | 0x8000)


def order_key16_bf16(bits: int) -> int:
    return (~bits) & 0xFFFF if (bits & 0x8000) else (bits | 0x8000)


# ---------------------------------------------------------------------------
# Capability negotiation mirrors.
# ---------------------------------------------------------------------------


def takes_bf16_logits(dtype_code: str, bits: int, lanes: int, stride0: int, vec_width: int) -> bool:
    return dtype_code == "bfloat" and bits == 16 and lanes == 1 and stride0 % vec_width == 0


def takes_fp32_logits(dtype_code: str, bits: int, lanes: int) -> bool:
    return dtype_code == "float" and bits == 32 and lanes == 1


def resolve_backend(dtype_code: str, bits: int, lanes: int, stride0: int) -> str:
    """Mirror of the dispatcher: BF16 when aligned, else FP32 upcast."""
    if takes_bf16_logits(dtype_code, bits, lanes, stride0, vec_width=8):
        return "bf16"
    if takes_fp32_logits(dtype_code, bits, lanes):
        return "fp32"
    return "upcast_fp32"


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "values",
    [
        [0.0, -0.0, 1.0, -1.0],
        [3.5, -2.25, 1e-3, -1e-3],
        [float("inf"), float("-inf"), 1.0, -1.0],
        [0.5, 0.25, 0.125, 0.0625],
    ],
)
def test_bf16x2_packed_bins_match_scalar(values):
    """Packed BF16x2 extraction must equal the scalar BF16 key."""
    bits = [fp32_to_bf16_bits(v) for v in values]
    for i in range(0, len(bits), 2):
        w = bits[i] | (bits[i + 1] << 16)
        lo, hi = convert_bf16x2_to_uint8x2(w)
        assert lo == convert_bf16_to_uint8(bits[i])
        assert hi == convert_bf16_to_uint8(bits[i + 1])


def test_bf16_key_is_monotonic():
    """The ordered key must be monotonic in the BF16 value."""
    rng = np.random.default_rng(0)
    vals = np.sort(rng.standard_normal(4096).astype(np.float32))
    keys = [convert_bf16_to_uint8(fp32_to_bf16_bits(float(v))) for v in vals]
    assert keys == sorted(keys)


def test_bf16_and_fp32_select_same_topk():
    """BF16 widening is exact, so top-k selection must agree with FP32."""
    rng = np.random.default_rng(1)
    row = rng.standard_normal(8192).astype(np.float32)
    # Values that survive BF16 rounding exactly.
    row = np.round(row * 8.0) / 8.0
    k = 512
    fp32_idx = np.argsort(-row, kind="stable")[:k]
    bf16 = np.array([bf16_bits_to_float(fp32_to_bf16_bits(float(v))) for v in row], dtype=np.float32)
    bf16_idx = np.argsort(-bf16, kind="stable")[:k]
    np.testing.assert_array_equal(np.sort(fp32_idx), np.sort(bf16_idx))
    np.testing.assert_array_equal(row[fp32_idx], bf16[bf16_idx])


def test_bf16_histogram_bins_match_fp32_bins():
    """Coarse histogram bins from BF16x2 must match the FP32 scalar bins."""
    rng = np.random.default_rng(2)
    row = rng.standard_normal(4096).astype(np.float32)
    row = np.round(row * 16.0) / 16.0
    hist_bf16 = np.zeros(256, dtype=np.int64)
    hist_fp32 = np.zeros(256, dtype=np.int64)
    for i in range(0, len(row), 2):
        w = fp32_to_bf16_bits(float(row[i])) | (fp32_to_bf16_bits(float(row[i + 1])) << 16)
        lo, hi = convert_bf16x2_to_uint8x2(w)
        hist_bf16[lo] += 1
        hist_bf16[hi] += 1
    for v in row:
        hist_fp32[convert_to_uint8_fp32(float(v))] += 1
    np.testing.assert_array_equal(hist_bf16, hist_fp32)


@pytest.mark.parametrize(
    "dtype_code,bits,lanes,stride0,expected",
    [
        ("bfloat", 16, 1, 8192, "bf16"),
        ("bfloat", 16, 1, 8193, "upcast_fp32"),
        ("float", 32, 1, 8193, "fp32"),
        ("float", 32, 1, 8192, "fp32"),
        ("bfloat", 16, 2, 8192, "upcast_fp32"),
    ],
)
def test_backend_dtype_negotiation(dtype_code, bits, lanes, stride0, expected):
    assert resolve_backend(dtype_code, bits, lanes, stride0) == expected


def test_bf16_fallback_when_unaligned():
    """Unaligned BF16 rows must fall back to the FP32 upcast path."""
    assert not takes_bf16_logits("bfloat", 16, 1, 8191, vec_width=8)
    assert resolve_backend("bfloat", 16, 1, 8191) == "upcast_fp32"


def test_gfx950_order_key16_matches_bf16_key():
    """gfx950 order_key16 on FP32 must agree with the BF16 ordered key."""
    rng = np.random.default_rng(3)
    for v in rng.standard_normal(1024).astype(np.float32):
        # Values exactly representable in both FP16 and BF16.
        v = float(np.float16(v))
        assert order_key16_fp32(v) == order_key16_bf16(fp32_to_bf16_bits(v))
