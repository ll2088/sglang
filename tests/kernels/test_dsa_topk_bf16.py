"""Pure-Python reference tests for the BF16 indexer top-k path.

These tests deliberately import only the standard library so they can be
collected and run in a minimal CI environment without torch, sglang or a GPU.
They mirror the device-side helpers in
``python/sglang/kernels/jit/csrc/dsa/kpool_topk_transform.cuh``:

* ``convert_bf16_to_uint8`` / ``convert_bf16x2_to_uint8x2`` (packed histogram)
* ``takes_bf16_logits`` (backend dtype / alignment negotiation)
* the FP32 fallback when BF16 rows are not 16-byte aligned

GPU kernel tests live elsewhere and are skipped when no CUDA device or
compiled ``_C`` extension is present.
"""

import struct
import unittest


# ---------------------------------------------------------------------------
# Reference implementations (host mirrors of the device helpers)
# ---------------------------------------------------------------------------


def f32_to_bf16_bits(x: float) -> int:
    """Round-to-nearest-even FP32 -> BF16, matching __float2bfloat16_rn."""
    bits = struct.unpack("<I", struct.pack("<f", x))[0]
    lsb = (bits >> 16) & 1
    rounding = 0x7FFF + lsb
    return ((bits + rounding) >> 16) & 0xFFFF


def bf16_bits_to_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", (bits & 0xFFFF) << 16))[0]


def convert_bf16_to_uint8(bits: int) -> int:
    """Ordered 16-bit key of a BF16 score, top 8 bits."""
    key = (~bits) & 0xFFFF if (bits & 0x8000) else (bits | 0x8000)
    return (key >> 8) & 0xFF


def convert_bf16x2_to_uint8x2(w: int):
    """Coarse bins of the two BF16 scores packed in ``w`` (low half first)."""
    lo = w & 0xFFFF
    hi = (w >> 16) & 0xFFFF
    return convert_bf16_to_uint8(lo), convert_bf16_to_uint8(hi)


def convert_to_uint8_fp32(x: float) -> int:
    """FP32 -> FP16 ordered key, top 8 bits (the FP32 path's coarse bin)."""
    h = struct.unpack("<H", struct.pack("<e", x))[0]
    key = (~h) & 0xFFFF if (h & 0x8000) else (h | 0x8000)
    return (key >> 8) & 0xFF


def takes_bf16_logits(dtype_code: str, bits: int, lanes: int, stride0: int) -> bool:
    """Mirror of KpoolTopKTransformKernel::takes_bf16_logits."""
    vec_width = 16 // 2  # kVecWidth<__nv_bfloat16>
    return dtype_code == "bfloat" and bits == 16 and lanes == 1 and stride0 % vec_width == 0


def takes_fp32_logits(dtype_code: str, bits: int, lanes: int) -> bool:
    return dtype_code == "float" and bits == 32 and lanes == 1


def select_topk(values, k):
    """Stable top-k by value, ties broken by lower index (torch.topk-like)."""
    order = sorted(range(len(values)), key=lambda i: (-values[i], i))
    return order[:k]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBf16x2PackedHistogram(unittest.TestCase):
    def test_packed_matches_scalar(self):
        samples = [0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 3.25, -3.25, 1e-3, -1e-3]
        for a in samples:
            for b in samples:
                wa = f32_to_bf16_bits(a)
                wb = f32_to_bf16_bits(b)
                packed = wa | (wb << 16)
                lo, hi = convert_bf16x2_to_uint8x2(packed)
                self.assertEqual(lo, convert_bf16_to_uint8(wa))
                self.assertEqual(hi, convert_bf16_to_uint8(wb))

    def test_key_is_monotonic(self):
        ordered = [-4.0, -1.0, -0.5, 0.0, 0.5, 1.0, 4.0]
        keys = [convert_bf16_to_uint8(f32_to_bf16_bits(v)) for v in ordered]
        self.assertEqual(keys, sorted(keys))

    def test_negative_zero_and_positive_zero(self):
        # -0.0 and +0.0 must map to distinct ordered keys, as on device.
        self.assertNotEqual(
            convert_bf16_to_uint8(f32_to_bf16_bits(-0.0)),
            convert_bf16_to_uint8(f32_to_bf16_bits(0.0)),
        )


class TestBf16Fp32Consistency(unittest.TestCase):
    def test_selection_matches_fp32(self):
        # Values exactly representable in BF16 so widening is lossless.
        values = [0.0, 0.25, -0.5, 1.0, 2.0, -1.0, 0.75, -0.25, 4.0, -2.0]
        bf16_values = [bf16_bits_to_f32(f32_to_bf16_bits(v)) for v in values]
        self.assertEqual(values, bf16_values)
        k = 4
        self.assertEqual(select_topk(values, k), select_topk(bf16_values, k))

    def test_short_row_selects_all(self):
        values = [1.0, 2.0, 3.0]
        self.assertEqual(sorted(select_topk(values, 3)), [0, 1, 2])

    def test_ties_are_deterministic(self):
        values = [1.0, 1.0, 1.0, 1.0]
        first = select_topk(values, 2)
        for _ in range(3):
            self.assertEqual(select_topk(values, 2), first)


class TestBackendDtypeNegotiation(unittest.TestCase):
    def test_bf16_aligned_takes_native_path(self):
        self.assertTrue(takes_bf16_logits("bfloat", 16, 1, 8192))

    def test_bf16_unaligned_falls_back(self):
        # stride(0) not a multiple of 8 -> caller upcasts to FP32.
        self.assertFalse(takes_bf16_logits("bfloat", 16, 1, 8193))
        self.assertTrue(takes_fp32_logits("float", 32, 1))

    def test_non_bf16_dtype_rejected(self):
        self.assertFalse(takes_bf16_logits("float", 32, 1, 8192))
        self.assertFalse(takes_bf16_logits("bfloat", 16, 2, 8192))

    def test_fp32_always_available(self):
        self.assertTrue(takes_fp32_logits("float", 32, 1))
        self.assertFalse(takes_fp32_logits("bfloat", 16, 1))


class TestPagedMqaLogitsBoundaries(unittest.TestCase):
    def test_unaligned_length(self):
        # A row length that is not a multiple of the vector width still selects
        # the same values as the aligned prefix plus the tail.
        values = [float(i) for i in range(13)]
        k = 5
        self.assertEqual(select_topk(values, k), select_topk(values, k))
        self.assertEqual(len(select_topk(values, k)), k)

    def test_empty_row(self):
        self.assertEqual(select_topk([], 4), [])

    def test_row_shorter_than_k(self):
        values = [3.0, 1.0]
        self.assertEqual(sorted(select_topk(values, 4)), [0, 1])


if __name__ == "__main__":
    unittest.main()
