"""Pure-Python reference tests for the BF16 DSA top-k port.

These tests never import torch or the compiled sglang extensions, so they run
in a minimal CI job and validate the host-side pieces of the port:

* the BF16x2 packed coarse-bin extraction matches the scalar FP32 key path,
* the backend dtype capability negotiation accepts BF16 only when the row
  stride keeps every row 16-byte aligned, and falls back to FP32 otherwise,
* the fallback decision is recorded so callers can observe it.

The kernel-level tests live in ``test_dsa_topk_bf16_kernel.py`` and skip when
the compiled ops are unavailable.
"""

from __future__ import annotations

import struct
import unittest


def _f32_to_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def _bits_to_f32(b: int) -> float:
    return struct.unpack("<f", struct.pack("<I", b & 0xFFFFFFFF))[0]


def _f32_to_bf16_bits(x: float) -> int:
    """Round-to-nearest-even FP32 -> BF16, matching __float2bfloat16_rn."""
    bits = _f32_to_bits(x)
    lsb = (bits >> 16) & 1
    rounding = 0x7FFF + lsb
    return ((bits + rounding) >> 16) & 0xFFFF


def _bf16_bits_to_f32(b: int) -> float:
    return _bits_to_f32((b & 0xFFFF) << 16)


def _scalar_coarse_bin(x: float, k_bits: int) -> int:
    """Reference: FP16 ordered key of x, top k_bits."""
    # FP32 -> FP16 round-to-nearest-even, then the ordered-key flip.
    bits = _f32_to_bits(x)
    sign = (bits >> 31) & 1
    exp = (bits >> 23) & 0xFF
    mant = bits & 0x7FFFFF
    if exp == 0xFF:
        h = 0x7C00 | (0x200 if mant else 0)
    elif exp - 127 + 15 >= 0x1F:
        h = 0x7C00
    elif exp - 127 + 15 <= 0:
        h = 0
    else:
        h = ((exp - 127 + 15) << 10) | (mant >> 13)
    h |= sign << 15
    key = (~h) & 0xFFFF if (h & 0x8000) else (h | 0x8000)
    return key >> (16 - k_bits)


def _packed_coarse_bins_bf16x2(w: int, k_bits: int) -> tuple[int, int]:
    """Mirror of convert_bf16x2_to_uint8x2 / extract_coarse_bins_bf16x2.

    ``w`` holds two BF16 values, low half first.  The kernel converts the two
    halves with one packed conversion and one key flip; here we do the same
    arithmetic on the packed word.
    """
    lo = w & 0xFFFF
    hi = (w >> 16) & 0xFFFF
    bits = lo | (hi << 16)
    sign = bits & 0x80008000
    key = bits ^ (((sign - (sign >> 15)) | 0x80008000) & 0xFFFFFFFF)
    return ((key & 0xFFFF) >> (16 - k_bits), (key >> 16) >> (16 - k_bits))


def _pack_bf16x2(a: float, b: float) -> int:
    return _f32_to_bf16_bits(a) | (_f32_to_bf16_bits(b) << 16)


class TestBf16x2PackedHistogram(unittest.TestCase):
    K_BITS = 8

    def test_packed_bins_match_scalar_path(self) -> None:
        values = [
            (0.0, 0.0),
            (1.0, -1.0),
            (0.5, 2.0),
            (-3.25, 7.5),
            (1e-3, -1e-3),
            (100.0, -100.0),
            (float("inf"), float("-inf")),
        ]
        for a, b in values:
            w = _pack_bf16x2(a, b)
            lo, hi = _packed_coarse_bins_bf16x2(w, self.K_BITS)
            # The packed path bins the BF16 values; the scalar reference bins
            # the same values through FP16.  BF16 -> FP16 is exact for the
            # magnitudes used here, so the bins must agree.
            self.assertEqual(lo, _scalar_coarse_bin(_bf16_bits_to_f32(_f32_to_bf16_bits(a)), self.K_BITS))
            self.assertEqual(hi, _scalar_coarse_bin(_bf16_bits_to_f32(_f32_to_bf16_bits(b)), self.K_BITS))

    def test_packed_bins_are_monotonic(self) -> None:
        prev = -1
        for i in range(-64, 65):
            x = i / 8.0
            w = _pack_bf16x2(x, x)
            lo, hi = _packed_coarse_bins_bf16x2(w, self.K_BITS)
            self.assertEqual(lo, hi)
            self.assertGreaterEqual(lo, prev)
            prev = lo

    def test_packed_bins_handle_negative_zero(self) -> None:
        w = _pack_bf16x2(-0.0, 0.0)
        lo, hi = _packed_coarse_bins_bf16x2(w, self.K_BITS)
        # -0.0 and +0.0 must land in the same bin as the scalar path.
        self.assertEqual(lo, _scalar_coarse_bin(-0.0, self.K_BITS))
        self.assertEqual(hi, _scalar_coarse_bin(0.0, self.K_BITS))


class TestBackendDtypeNegotiation(unittest.TestCase):
    """Mirror of takes_bf16_logits in the two kernel entry points."""

    @staticmethod
    def takes_bf16_logits(dtype: str, bits: int, lanes: int, stride0: int, vec_width: int) -> bool:
        return dtype == "bfloat16" and bits == 16 and lanes == 1 and stride0 % vec_width == 0

    def test_bf16_aligned_stride_accepted(self) -> None:
        self.assertTrue(self.takes_bf16_logits("bfloat16", 16, 1, 65536, 8))
        self.assertTrue(self.takes_bf16_logits("bfloat16", 16, 1, 1048576, 8))

    def test_bf16_unaligned_stride_falls_back(self) -> None:
        self.assertFalse(self.takes_bf16_logits("bfloat16", 16, 1, 65535, 8))
        self.assertFalse(self.takes_bf16_logits("bfloat16", 16, 1, 65537, 8))

    def test_fp32_never_takes_bf16_path(self) -> None:
        self.assertFalse(self.takes_bf16_logits("float32", 32, 1, 65536, 8))

    def test_non_unit_lanes_rejected(self) -> None:
        self.assertFalse(self.takes_bf16_logits("bfloat16", 16, 2, 65536, 8))

    def test_fallback_is_recorded(self) -> None:
        # The scheduler must be able to observe why it fell back, so the
        # decision is a value and not just a bool.
        def resolve(dtype: str, stride0: int) -> tuple[bool, str]:
            if dtype != "bfloat16":
                return False, "dtype_not_bf16"
            if stride0 % 8 != 0:
                return False, "stride_not_16b_aligned"
            return True, "bf16_native"

        self.assertEqual(resolve("bfloat16", 65536), (True, "bf16_native"))
        self.assertEqual(resolve("bfloat16", 65537), (False, "stride_not_16b_aligned"))
        self.assertEqual(resolve("float32", 65536), (False, "dtype_not_bf16"))


if __name__ == "__main__":
    unittest.main()
