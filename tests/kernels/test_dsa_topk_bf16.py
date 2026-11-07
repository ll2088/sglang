"""Pure-Python BF16 top-k reference tests.

These tests never import torch or sglang, so they run in a minimal CI job and
cover the BF16x2 packed histogram key extraction that the CUDA kernels rely on.
The kernel-level tests live in ``test_dsa_topk_bf16_kernel.py`` and skip when
the compiled ops are unavailable.
"""

from __future__ import annotations

import struct
import unittest


def _bf16_bits(x: float) -> int:
    """Round a float to BF16 (round-to-nearest-even) and return its 16 bits."""
    (f32,) = struct.unpack("<I", struct.pack("<f", x))
    # Round-to-nearest-even on the low 16 bits.
    lsb = (f32 >> 16) & 1
    rounded = f32 + 0x7FFF + lsb
    return (rounded >> 16) & 0xFFFF


def _bf16_to_float(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def _order_key16(bits: int) -> int:
    """Ordered 16-bit key: ~bits if negative, else bits | 0x8000."""
    if bits & 0x8000:
        return (~bits) & 0xFFFF
    return bits | 0x8000


def _coarse_bin(bits: int, k_bits: int) -> int:
    return _order_key16(bits) >> (16 - k_bits)


def _packed_bins_bf16x2(w: int, k_bits: int) -> tuple[int, int]:
    """Reference of extract_coarse_bins_bf16x2: low half first."""
    lo = w & 0xFFFF
    hi = (w >> 16) & 0xFFFF
    return _coarse_bin(lo, k_bits), _coarse_bin(hi, k_bits)


def _pack_bf16x2(a: float, b: float) -> int:
    return _bf16_bits(a) | (_bf16_bits(b) << 16)


class TestBf16x2PackedBins(unittest.TestCase):
    def test_matches_scalar_bins(self) -> None:
        values = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            0.5,
            -0.5,
            3.25,
            -3.25,
            1e-3,
            -1e-3,
            1e3,
            -1e3,
        ]
        for k_bits in (8, 12):
            for a in values:
                for b in values:
                    w = _pack_bf16x2(a, b)
                    lo, hi = _packed_bins_bf16x2(w, k_bits)
                    self.assertEqual(lo, _coarse_bin(_bf16_bits(a), k_bits))
                    self.assertEqual(hi, _coarse_bin(_bf16_bits(b), k_bits))

    def test_ordering_is_monotonic(self) -> None:
        # The ordered key must preserve numeric order, which is what makes the
        # histogram threshold sound.
        values = [-4.0, -1.0, -0.0, 0.0, 0.25, 1.0, 4.0]
        keys = [_order_key16(_bf16_bits(v)) for v in values]
        self.assertEqual(keys, sorted(keys))

    def test_negative_zero_and_zero_share_bin(self) -> None:
        for k_bits in (8, 12):
            self.assertEqual(
                _coarse_bin(_bf16_bits(0.0), k_bits),
                _coarse_bin(_bf16_bits(-0.0), k_bits),
            )

    def test_round_trip(self) -> None:
        for v in (0.0, 1.0, -1.0, 2.5, -2.5):
            self.assertEqual(_bf16_to_float(_bf16_bits(v)), v)


if __name__ == "__main__":
    unittest.main()
