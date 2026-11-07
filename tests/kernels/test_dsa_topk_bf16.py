"""Pure-Python BF16 top-k reference tests.

This module deliberately imports only the standard library so that a minimal
CI job (pytest, no torch, no sglang) can collect and run it.  It mirrors the
BF16x2 packed histogram extraction used by the DSA top-k kernels and checks
that the packed path selects the same coarse bins as the scalar FP32 path.

The GPU kernel tests live in ``test_dsa_topk_bf16_kernel.py`` and are opt-in
via ``SGLANG_RUN_GPU_KERNEL_TESTS=1``.
"""

from __future__ import annotations

import struct
import unittest


def _f32_to_bf16_bits(x: float) -> int:
    """Round-to-nearest-even FP32 -> BF16, matching __float2bfloat16_rn."""
    bits = struct.unpack("<I", struct.pack("<f", x))[0]
    # NaN stays NaN; infinities map to infinities.
    if (bits & 0x7F800000) == 0x7F800000 and (bits & 0x007FFFFF) != 0:
        return (bits >> 16) | 0x0040
    lsb = (bits >> 16) & 1
    rounding_bias = 0x7FFF + lsb
    return ((bits + rounding_bias) >> 16) & 0xFFFF


def _bf16_bits_to_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", (bits & 0xFFFF) << 16))[0]


def _order_key16_from_bf16_bits(bits: int) -> int:
    """Ordered 16-bit key of a BF16 value: ~bits if negative, else bits | 0x8000."""
    if bits & 0x8000:
        return (~bits) & 0xFFFF
    return bits | 0x8000


def _coarse_bin_scalar(x: float, k_bits: int) -> int:
    """Reference: BF16-round then take the top k_bits of the ordered key."""
    key = _order_key16_from_bf16_bits(_f32_to_bf16_bits(x))
    return key >> (16 - k_bits)


def _coarse_bins_bf16x2(w: int, k_bits: int) -> tuple[int, int]:
    """Packed extraction of the two BF16 coarse bins in a 32-bit word.

    ``w`` holds two BF16 values, low half first.  This mirrors
    ``extract_coarse_bins_bf16x2`` in the kernel: one packed conversion and
    key flip, then a shift per half.
    """
    lo = w & 0xFFFF
    hi = (w >> 16) & 0xFFFF
    key_lo = _order_key16_from_bf16_bits(lo)
    key_hi = _order_key16_from_bf16_bits(hi)
    return key_lo >> (16 - k_bits), key_hi >> (16 - k_bits)


def _pack_bf16x2(a: float, b: float) -> int:
    return (_f32_to_bf16_bits(b) << 16) | _f32_to_bf16_bits(a)


def _topk_indices(values: list[float], k: int) -> list[int]:
    """Stable descending top-k by value, ties broken by lower index."""
    order = sorted(range(len(values)), key=lambda i: (-values[i], i))
    return order[:k]


class TestBf16x2CoarseBins(unittest.TestCase):
    def test_packed_matches_scalar(self) -> None:
        samples = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            0.5,
            -0.5,
            3.14159,
            -2.71828,
            1e-8,
            -1e-8,
            1e8,
            -1e8,
            float("inf"),
            float("-inf"),
        ]
        for k_bits in (8, 10, 12):
            for i in range(0, len(samples) - 1, 2):
                a, b = samples[i], samples[i + 1]
                w = _pack_bf16x2(a, b)
                packed = _coarse_bins_bf16x2(w, k_bits)
                scalar = (_coarse_bin_scalar(a, k_bits), _coarse_bin_scalar(b, k_bits))
                self.assertEqual(packed, scalar, msg=f"k_bits={k_bits} a={a} b={b}")

    def test_packed_ordering_is_monotonic(self) -> None:
        # The ordered key must be monotonic in the BF16 value so that a
        # threshold bin separates winners from losers.
        values = [-4.0, -1.0, -0.25, 0.0, 0.25, 1.0, 4.0]
        keys = [_order_key16_from_bf16_bits(_f32_to_bf16_bits(v)) for v in values]
        self.assertEqual(keys, sorted(keys))

    def test_topk_selection_matches_reference(self) -> None:
        # Build a small row, bin it with the packed path, and check that the
        # selected indices agree with a plain top-k over the same values.
        values = [((i * 37) % 101 - 50) / 7.0 for i in range(64)]
        k = 8
        k_bits = 8
        bins = []
        for i in range(0, len(values), 2):
            w = _pack_bf16x2(values[i], values[i + 1])
            bins.extend(_coarse_bins_bf16x2(w, k_bits))
        self.assertEqual(len(bins), len(values))
        # The threshold bin is the one containing the k-th largest value.
        sorted_bins = sorted(bins, reverse=True)
        threshold = sorted_bins[k - 1]
        candidates = [i for i, b in enumerate(bins) if b >= threshold]
        self.assertGreaterEqual(len(candidates), k)
        # Refine within the candidate set using the exact values.
        refined = sorted(candidates, key=lambda i: (-values[i], i))[:k]
        expected = _topk_indices(values, k)
        self.assertEqual(refined, expected)

    def test_negative_zero_and_inf(self) -> None:
        # -0.0 and +0.0 must not collide in a way that breaks the key order.
        key_pos = _order_key16_from_bf16_bits(_f32_to_bf16_bits(0.0))
        key_neg = _order_key16_from_bf16_bits(_f32_to_bf16_bits(-0.0))
        self.assertLessEqual(key_neg, key_pos)
        key_inf = _order_key16_from_bf16_bits(_f32_to_bf16_bits(float("inf")))
        key_ninf = _order_key16_from_bf16_bits(_f32_to_bf16_bits(float("-inf")))
        self.assertLess(key_ninf, key_pos)
        self.assertLess(key_pos, key_inf)


class TestBf16RoundTrip(unittest.TestCase):
    def test_round_trip_exact_for_bf16_values(self) -> None:
        for bits in (0x0000, 0x3F80, 0xBF80, 0x7F7F, 0xFF7F):
            x = _bf16_bits_to_f32(bits)
            self.assertEqual(_f32_to_bf16_bits(x), bits)


if __name__ == "__main__":
    unittest.main()
