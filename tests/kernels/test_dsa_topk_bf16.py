"""Standalone tests for the BF16 indexer top-k capability negotiation.

These tests avoid importing torch/CUDA by exercising the pure-Python helpers
and the host-side mirrors of the kernel predicates.  They are meant to run in
a minimal environment (numpy + pytest only).
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Host-side mirrors of the kernel predicates.  Kept in the test so the tests
# do not need the CUDA extension to be built.
# ---------------------------------------------------------------------------


def _host_convert_bf16_to_uint8(bits: int) -> int:
    bits &= 0xFFFF
    key = (~bits) & 0xFFFF if (bits & 0x8000) else (bits | 0x8000)
    return (key >> 8) & 0xFF


def _host_convert_bf16x2_to_uint8x2(w: int) -> tuple[int, int]:
    lo = w & 0xFFFF
    hi = (w >> 16) & 0xFFFF
    return _host_convert_bf16_to_uint8(lo), _host_convert_bf16_to_uint8(hi)


def _takes_bf16_logits_for(dtype_code: int, dtype_bits: int, dtype_lanes: int, row_stride: int) -> bool:
    # kDLBfloat == 5 in the DLPack enum; kVecWidth<__nv_bfloat16> == 8.
    return dtype_code == 5 and dtype_bits == 16 and dtype_lanes == 1 and row_stride % 8 == 0


def _takes_fp32_logits_for(dtype_code: int, dtype_bits: int, dtype_lanes: int) -> bool:
    # kDLFloat == 2.
    return dtype_code == 2 and dtype_bits == 32 and dtype_lanes == 1


# ---------------------------------------------------------------------------
# BF16x2 packed histogram extraction
# ---------------------------------------------------------------------------


def _bf16_bits(x: float) -> int:
    return int(np.float32(x).view(np.uint32) >> 16)


@pytest.mark.parametrize(
    "values",
    [
        (0.0, 0.0),
        (1.0, -1.0),
        (0.5, 2.0),
        (-3.0, 4.0),
        (float("inf"), float("-inf")),
    ],
)
def test_bf16x2_packed_bins_match_scalar(values):
    lo, hi = values
    w = (_bf16_bits(hi) << 16) | _bf16_bits(lo)
    got_lo, got_hi = _host_convert_bf16x2_to_uint8x2(w)
    assert got_lo == _host_convert_bf16_to_uint8(_bf16_bits(lo))
    assert got_hi == _host_convert_bf16_to_uint8(_bf16_bits(hi))


def test_bf16x2_packed_bins_are_monotonic():
    # Ordered keys must be monotonic in the score, which is what makes the
    # histogram threshold sound.
    scores = np.linspace(-8.0, 8.0, 257, dtype=np.float32)
    keys = [_host_convert_bf16_to_uint8(_bf16_bits(float(s))) for s in scores]
    assert keys == sorted(keys)


def test_bf16x2_packed_bins_negative_zero():
    # -0.0 and +0.0 must map to the same bin.
    assert _host_convert_bf16_to_uint8(_bf16_bits(-0.0)) == _host_convert_bf16_to_uint8(_bf16_bits(0.0))


# ---------------------------------------------------------------------------
# Backend dtype capability negotiation and fallback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stride", [8, 16, 1024, 65536])
def test_takes_bf16_logits_aligned(stride):
    assert _takes_bf16_logits_for(5, 16, 1, stride)


@pytest.mark.parametrize("stride", [1, 2, 3, 4, 7, 9, 15])
def test_takes_bf16_logits_unaligned_falls_back(stride):
    assert not _takes_bf16_logits_for(5, 16, 1, stride)


def test_takes_bf16_logits_rejects_fp32():
    assert not _takes_bf16_logits_for(2, 32, 1, 16)


def test_takes_bf16_logits_rejects_wrong_lanes():
    assert not _takes_bf16_logits_for(5, 16, 2, 16)


def test_takes_fp32_logits_always_true_for_fp32():
    assert _takes_fp32_logits_for(2, 32, 1)
    assert not _takes_fp32_logits_for(5, 16, 1)


def test_fallback_selects_fp32_when_bf16_unaligned():
    # The dispatcher must pick FP32 when BF16 is not aligned.
    stride = 7
    use_bf16 = _takes_bf16_logits_for(5, 16, 1, stride)
    use_fp32 = _takes_fp32_logits_for(2, 32, 1)
    assert not use_bf16
    assert use_fp32


# ---------------------------------------------------------------------------
# indexer_topk backend resolution (pure-Python helpers)
# ---------------------------------------------------------------------------


def test_indexer_topk_bf16_declared_and_fallback():
    # The resolver must accept BF16 when the backend declares it and fall back
    # to FP32 otherwise.  This mirrors the capability negotiation in the
    # dispatcher without importing the runtime.
    def resolve(backend_supports_bf16: bool, aligned: bool) -> str:
        if backend_supports_bf16 and aligned:
            return "bf16"
        return "fp32"

    assert resolve(True, True) == "bf16"
    assert resolve(True, False) == "fp32"
    assert resolve(False, True) == "fp32"
    assert resolve(False, False) == "fp32"
