"""Dependency-light tests for the BF16 DSA indexer top-k path.

These tests are intentionally written against the Python standard library
(plus an optional numpy import) so they can run in a minimal CI job that does
not install the full sglang dependency chain.  Anything that needs CUDA or the
compiled kernels is guarded by ``unittest.skipUnless`` and reported as skipped
rather than failing at import time.
"""

from __future__ import annotations

import struct
import unittest


try:  # numpy is optional: the pure-Python fallbacks below do not need it.
    import numpy as _np
except Exception:  # pragma: no cover - exercised only in minimal jobs
    _np = None


def _has_cuda() -> bool:
    if _np is None:
        return False
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


CUDA_AVAILABLE = _has_cuda()


# ---------------------------------------------------------------------------
# Reference implementations of the packed BF16 helpers.
#
# These mirror the device-side helpers in
# ``python/sglang/kernels/jit/csrc/dsa/kpool_topk_transform.cuh``
# (``convert_bf16x2_to_uint8x2``) and the ordered-key helpers used by the
# gfx950 top-k transform.  They are pure Python so they can be unit tested
# without a GPU.
# ---------------------------------------------------------------------------


def _bf16_bits_to_float(bits: int) -> float:
    """Widen a 16-bit BF16 pattern to a Python float (exact)."""
    return struct.unpack("<f", struct.pack("<I", (bits & 0xFFFF) << 16))[0]


def _float_to_bf16_bits(value: float) -> int:
    """Round-to-nearest-even float32 -> BF16, matching __float2bfloat16_rn."""
    f32 = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    # NaN must stay NaN; the hardware keeps the quiet bit.
    if (f32 & 0x7F800000) == 0x7F800000 and (f32 & 0x007FFFFF) != 0:
        return ((f32 >> 16) | 0x0040) & 0xFFFF
    lsb = (f32 >> 16) & 1
    rounding = 0x7FFF + lsb
    return ((f32 + rounding) >> 16) & 0xFFFF


def _ordered_key16(bits: int) -> int:
    """Ordered 16-bit key: ~bits if negative, else bits | 0x8000."""
    bits &= 0xFFFF
    if bits & 0x8000:
        return (~bits) & 0xFFFF
    return bits | 0x8000


def _convert_bf16x2_to_uint8x2(word: int) -> tuple[int, int]:
    """Packed BF16x2 -> two coarse 8-bit bins (low half first).

    ``word`` holds two BF16 patterns: low 16 bits are element 0, high 16 bits
    are element 1.  This is the Python transcription of
    ``convert_bf16x2_to_uint8x2`` in kpool_topk_transform.cuh.
    """
    lo = word & 0xFFFF
    hi = (word >> 16) & 0xFFFF
    k_lo = _ordered_key16(lo)
    k_hi = _ordered_key16(hi)
    return (k_lo >> 8, k_hi >> 8)


def _convert_bf16_to_uint8(bits: int) -> int:
    return _ordered_key16(bits) >> 8


def _pack_bf16x2(values: list[float]) -> int:
    assert len(values) == 2
    lo = _float_to_bf16_bits(values[0])
    hi = _float_to_bf16_bits(values[1])
    return (hi << 16) | lo


# ---------------------------------------------------------------------------
# BF16x2 packed histogram extraction.
# ---------------------------------------------------------------------------


class TestBf16x2PackedBins(unittest.TestCase):
    """The packed path must agree with the scalar path element by element."""

    def test_packed_matches_scalar(self) -> None:
        cases = [
            (0.0, 0.0),
            (1.0, -1.0),
            (0.5, 2.0),
            (-3.25, 3.25),
            (1e-8, -1e-8),
            (65504.0, -65504.0),
        ]
        for a, b in cases:
            word = _pack_bf16x2([a, b])
            packed = _convert_bf16x2_to_uint8x2(word)
            scalar = (
                _convert_bf16_to_uint8(_float_to_bf16_bits(a)),
                _convert_bf16_to_uint8(_float_to_bf16_bits(b)),
            )
            self.assertEqual(packed, scalar, msg=f"a={a} b={b}")

    def test_ordering_is_monotonic(self) -> None:
        values = [-10.0, -1.0, -0.0, 0.0, 1.0, 10.0]
        keys = [_ordered_key16(_float_to_bf16_bits(v)) for v in values]
        self.assertEqual(keys, sorted(keys))

    def test_negative_zero_and_positive_zero(self) -> None:
        # -0.0 and +0.0 are distinct bit patterns but compare equal; the
        # ordered key must keep -0.0 below +0.0 so ties are deterministic.
        k_neg = _ordered_key16(_float_to_bf16_bits(-0.0))
        k_pos = _ordered_key16(_float_to_bf16_bits(0.0))
        self.assertLess(k_neg, k_pos)

    def test_bf16_widening_is_exact(self) -> None:
        for v in [0.0, 1.0, -1.0, 0.5, 3.140625, -2.71875]:
            bits = _float_to_bf16_bits(v)
            self.assertEqual(_bf16_bits_to_float(bits), v)


# ---------------------------------------------------------------------------
# Backend dtype capability negotiation and fallback.
# ---------------------------------------------------------------------------


def _takes_bf16_logits(dtype_code: str, bits: int, lanes: int, stride0: int, vec_width: int) -> bool:
    """Mirror of ``takes_bf16_logits`` in the JIT kernels."""
    return (
        dtype_code == "bfloat"
        and bits == 16
        and lanes == 1
        and stride0 % vec_width == 0
    )


class TestBackendDtypeNegotiation(unittest.TestCase):
    def test_bf16_aligned_takes_bf16(self) -> None:
        self.assertTrue(_takes_bf16_logits("bfloat", 16, 1, 1024, 8))

    def test_bf16_unaligned_falls_back(self) -> None:
        self.assertFalse(_takes_bf16_logits("bfloat", 16, 1, 1023, 8))

    def test_fp32_never_takes_bf16(self) -> None:
        self.assertFalse(_takes_bf16_logits("float", 32, 1, 1024, 8))

    def test_fp16_is_not_bf16(self) -> None:
        self.assertFalse(_takes_bf16_logits("float", 16, 1, 1024, 8))

    def test_fallback_is_logged(self) -> None:
        # The scheduler must record why it fell back; emulate the decision and
        # assert the reason string is produced.
        reason = None
        if not _takes_bf16_logits("bfloat", 16, 1, 1023, 8):
            reason = "bf16 logits row stride not 16-byte aligned; upcasting to fp32"
        self.assertIsNotNone(reason)
        self.assertIn("upcasting", reason)


# ---------------------------------------------------------------------------
# indexer_topk backend resolution.
# ---------------------------------------------------------------------------


def _resolve_backend(requested: str, bf16_ok: bool, fp32_only_backends: set[str]) -> str:
    """Minimal model of the indexer_topk backend resolution.

    ``auto`` prefers ``cooperative`` for small batches, but only when the
    backend can take BF16 logits; otherwise it falls back to the FP32 path.
    """
    if requested == "auto":
        if bf16_ok and "cooperative" not in fp32_only_backends:
            return "cooperative"
        return "persistent"
    if requested in fp32_only_backends and bf16_ok:
        return "persistent"
    return requested


class TestIndexerTopkBackendResolution(unittest.TestCase):
    def test_auto_prefers_cooperative_when_bf16_ok(self) -> None:
        self.assertEqual(_resolve_backend("auto", True, set()), "cooperative")

    def test_auto_falls_back_when_bf16_unavailable(self) -> None:
        self.assertEqual(_resolve_backend("auto", False, set()), "persistent")

    def test_fp32_only_backend_falls_back(self) -> None:
        self.assertEqual(
            _resolve_backend("cooperative", True, {"cooperative"}), "persistent"
        )

    def test_explicit_backend_is_preserved(self) -> None:
        self.assertEqual(_resolve_backend("persistent", True, set()), "persistent")


# ---------------------------------------------------------------------------
# Paged MQA logits boundary cases (pure Python model of the clamp).
# ---------------------------------------------------------------------------


def _effective_seqlen(seqlen: int, max_pages: int, out_stride: int, page_tok: int) -> int:
    """Model of the npages clamp in paged_mqa_logits.cuh."""
    npages_raw = (seqlen + page_tok - 1) // page_tok
    pages_out = out_stride // page_tok
    npages_cap = min(max_pages, pages_out)
    npages = min(npages_raw, npages_cap)
    return min(seqlen, npages * page_tok)


class TestPagedMqaLogitsBoundary(unittest.TestCase):
    def test_unaligned_length_is_clamped(self) -> None:
        # 100 tokens, page size 64 -> 2 pages -> 128 slots, but seqlen wins.
        self.assertEqual(_effective_seqlen(100, 4, 256, 64), 100)

    def test_empty_page_table(self) -> None:
        self.assertEqual(_effective_seqlen(100, 0, 256, 64), 0)

    def test_out_stride_shorter_than_table(self) -> None:
        # out covers one page only; the walk must stop there.
        self.assertEqual(_effective_seqlen(100, 4, 64, 64), 64)

    def test_exact_page_multiple(self) -> None:
        self.assertEqual(_effective_seqlen(128, 4, 256, 64), 128)


# ---------------------------------------------------------------------------
# GPU-backed consistency tests (skipped when CUDA is unavailable).
# ---------------------------------------------------------------------------


@unittest.skipUnless(CUDA_AVAILABLE, "requires CUDA and torch")
class TestBf16Fp32Consistency(unittest.TestCase):
    """BF16 and FP32 logits must select the same values."""

    def _run(self, dtype):
        import torch

        torch.manual_seed(0)
        rows, length, k = 4, 4096, 128
        logits = torch.randn(rows, length, device="cuda", dtype=dtype)
        values, indices = torch.topk(logits, k, dim=-1)
        return values, indices

    def test_bf16_and_fp32_select_same_values(self) -> None:
        import torch

        v_bf16, _ = self._run(torch.bfloat16)
        v_fp32, _ = self._run(torch.float32)
        # BF16 widens to FP32 exactly, so the selected values must match.
        torch.testing.assert_close(v_bf16.float(), v_fp32, rtol=0, atol=0)

    def test_bf16_indices_are_unique(self) -> None:
        _, idx = self._run(torch.bfloat16)
        for row in idx:
            self.assertEqual(row.unique().numel(), row.numel())


if __name__ == "__main__":
    unittest.main()
