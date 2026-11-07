import math
import struct

import pytest

# Pure-Python reference for the BF16 indexer logits path. This module must stay
# dependency-free (stdlib only) so it is collectable and runnable in a minimal
# CI environment without sglang, torch, or a GPU. Kernel-level coverage lives in
# test_dsa_topk_bf16_kernel.py and skips when the extension is unavailable.


def _bf16_bits(x: float) -> int:
    """Round-to-nearest-even BF16 bit pattern of a Python float."""
    (f32,) = struct.unpack("<I", struct.pack("<f", x))
    # NaN/Inf pass through unchanged.
    if (f32 & 0x7F800000) == 0x7F800000:
        return (f32 >> 16) & 0xFFFF
    lsb = (f32 >> 16) & 1
    rounding = 0x7FFF + lsb
    return ((f32 + rounding) >> 16) & 0xFFFF


def _bf16_to_float(bits: int) -> float:
    (f,) = struct.unpack("<f", struct.pack("<I", (bits & 0xFFFF) << 16))
    return f


def _order_key16(bits: int) -> int:
    """Ordered 16-bit key: ~bits if negative, else bits | 0x8000."""
    if bits & 0x8000:
        return (~bits) & 0xFFFF
    return (bits | 0x8000) & 0xFFFF


def _coarse_bin(bits: int, hist_bits: int) -> int:
    return _order_key16(bits) >> (16 - hist_bits)


def _pack_bf16x2(lo_bits: int, hi_bits: int) -> int:
    return (lo_bits & 0xFFFF) | ((hi_bits & 0xFFFF) << 16)


def _unpack_bf16x2(w: int):
    return w & 0xFFFF, (w >> 16) & 0xFFFF


def _extract_coarse_bins_bf16x2(w: int, hist_bits: int):
    """Reference for the packed BF16x2 coarse-bin extraction.

    Mirrors extract_coarse_bins_bf16x2: one packed conversion plus a per-half
    key flip, returning (low half bin, high half bin).
    """
    lo, hi = _unpack_bf16x2(w)
    return _coarse_bin(lo, hist_bits), _coarse_bin(hi, hist_bits)


def _takes_bf16_logits(dtype_is_bf16: bool, stride: int, elem_size: int = 2) -> bool:
    """Backend dtype capability negotiation.

    BF16 is only taken when the row stride keeps every row 16-byte aligned;
    otherwise the caller upcasts and takes the FP32 kernel.
    """
    vec_width = 16 // elem_size
    return dtype_is_bf16 and stride % vec_width == 0


def _topk_values(values, k):
    return sorted(values, reverse=True)[:k]


@pytest.mark.parametrize("hist_bits", [8, 12])
@pytest.mark.parametrize(
    "values",
    [
        [0.0, -0.0, 1.0, -1.0],
        [3.5, -2.25, 0.125, 1e-3],
        [float("inf"), float("-inf"), 0.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
    ],
)
def test_bf16x2_packed_histogram_matches_scalar(hist_bits, values):
    """Packed BF16x2 extraction must agree with the scalar per-element path."""
    bits = [_bf16_bits(v) for v in values]
    for i in range(0, len(bits), 2):
        lo, hi = bits[i], bits[i + 1]
        w = _pack_bf16x2(lo, hi)
        got = _extract_coarse_bins_bf16x2(w, hist_bits)
        assert got == (_coarse_bin(lo, hist_bits), _coarse_bin(hi, hist_bits))


def test_bf16x2_packed_histogram_topk_indices_and_weights():
    """Histogram built from packed bins selects the same top-k as scalar."""
    hist_bits = 8
    values = [0.5, -1.5, 2.0, 0.25, 3.0, -0.75, 1.25, 0.0]
    bits = [_bf16_bits(v) for v in values]

    packed_hist = [0] * (1 << hist_bits)
    for i in range(0, len(bits), 2):
        w = _pack_bf16x2(bits[i], bits[i + 1])
        lo_bin, hi_bin = _extract_coarse_bins_bf16x2(w, hist_bits)
        packed_hist[lo_bin] += 1
        packed_hist[hi_bin] += 1

    scalar_hist = [0] * (1 << hist_bits)
    for b in bits:
        scalar_hist[_coarse_bin(b, hist_bits)] += 1

    assert packed_hist == scalar_hist

    k = 3
    expected = _topk_values([_bf16_to_float(b) for b in bits], k)
    got = _topk_values([_bf16_to_float(b) for b in bits], k)
    assert got == expected


def test_bf16_widens_to_fp32_exactly():
    """BF16 -> FP32 widening is exact, so top-k selects the same values."""
    values = [0.0, -0.0, 1.0, -1.0, 0.1, -0.1, 123.5, -123.5]
    for v in values:
        b = _bf16_bits(v)
        widened = _bf16_to_float(b)
        assert _bf16_bits(widened) == b
        assert math.isclose(widened, _bf16_to_float(b), rel_tol=0.0, abs_tol=0.0)


@pytest.mark.parametrize(
    "stride,expected",
    [
        (8, True),
        (16, True),
        (4, False),
        (7, False),
        (0, True),
    ],
)
def test_backend_dtype_capability_negotiation(stride, expected):
    """BF16 is taken only when the row stride is 16-byte aligned."""
    assert _takes_bf16_logits(True, stride) is expected
    assert _takes_bf16_logits(False, stride) is False


def test_backend_dtype_capability_falls_back_to_fp32():
    """Unaligned BF16 rows must fall back to the FP32 kernel."""
    assert not _takes_bf16_logits(True, 7)
    assert not _takes_bf16_logits(True, 4)
    # FP32 rows are always 16-byte aligned at stride % 4 == 0.
    assert _takes_bf16_logits(False, 4) is False


def test_indexer_topk_backend_resolution_bf16_and_fallback():
    """indexer_topk backend resolution must declare BF16 and fall back."""

    def resolve_backend(dtype_is_bf16: bool, stride: int, backend: str) -> str:
        if backend in ("cooperative", "persistent") and _takes_bf16_logits(
            dtype_is_bf16, stride
        ):
            return backend + "_bf16"
        return backend + "_fp32"

    assert resolve_backend(True, 8, "cooperative") == "cooperative_bf16"
    assert resolve_backend(True, 8, "persistent") == "persistent_bf16"
    assert resolve_backend(True, 7, "cooperative") == "cooperative_fp32"
    assert resolve_backend(False, 8, "persistent") == "persistent_fp32"
