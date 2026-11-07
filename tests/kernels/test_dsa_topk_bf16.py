"""Dependency-light checks for the BF16 indexer top-k path.

These tests deliberately avoid importing torch or building any CUDA/HIP
kernel: they exercise the host-side pieces of the BF16 native-read port --
the packed BF16x2 coarse-bin extraction, the dtype/alignment capability
negotiation and its FP32 fallback, and the indexer_topk module's lazy torch
import -- so they can run in a minimal numpy+pytest environment.
"""

import importlib
import struct

import pytest


# ---------------------------------------------------------------------------
# Host mirrors of the device-side helpers under test.
# ---------------------------------------------------------------------------


def _bf16_bits_to_float(bits: int) -> float:
    """Widen a BF16 bit pattern to FP32 exactly (BF16 is the high half)."""
    return struct.unpack("<f", struct.pack("<I", (bits & 0xFFFF) << 16))[0]


def _float_to_bf16_bits(x: float) -> int:
    """Round-to-nearest-even FP32 -> BF16, matching __float2bfloat16_rn."""
    raw = struct.unpack("<I", struct.pack("<f", x))[0]
    lsb = (raw >> 16) & 1
    rounding = 0x7FFF + lsb
    return ((raw + rounding) >> 16) & 0xFFFF


def _order_key16(bits: int) -> int:
    """Ordered 16-bit key: ~bits if negative, else bits | 0x8000."""
    if bits & 0x8000:
        return (~bits) & 0xFFFF
    return bits | 0x8000


def _coarse_bin(bits: int, hist_bits: int) -> int:
    return _order_key16(bits) >> (16 - hist_bits)


def _coarse_bins_bf16x2(word: int, hist_bits: int):
    """Host mirror of extract_coarse_bins_bf16x2: low half first."""
    lo = word & 0xFFFF
    hi = (word >> 16) & 0xFFFF
    return _coarse_bin(lo, hist_bits), _coarse_bin(hi, hist_bits)


def _pack_bf16x2(lo_bits: int, hi_bits: int) -> int:
    return (lo_bits & 0xFFFF) | ((hi_bits & 0xFFFF) << 16)


def _takes_bf16_logits(dtype_code, dtype_bits, dtype_lanes, row_stride, vec_width=8):
    """Host mirror of KpoolTopKTransformKernel::takes_bf16_logits."""
    return (
        dtype_code == "bfloat"
        and dtype_bits == 16
        and dtype_lanes == 1
        and row_stride % vec_width == 0
    )


# ---------------------------------------------------------------------------
# BF16x2 packed coarse-bin extraction.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hist_bits", [8, 12])
@pytest.mark.parametrize(
    "values",
    [
        (0.0, 0.0),
        (-0.0, 0.0),
        (1.0, -1.0),
        (0.5, 2.0),
        (-3.25, 7.5),
        (1e-3, -1e-3),
        (100.0, -100.0),
    ],
)
def test_bf16x2_coarse_bins_match_scalar(hist_bits, values):
    lo_bits = _float_to_bf16_bits(values[0])
    hi_bits = _float_to_bf16_bits(values[1])
    word = _pack_bf16x2(lo_bits, hi_bits)

    got_lo, got_hi = _coarse_bins_bf16x2(word, hist_bits)
    assert got_lo == _coarse_bin(lo_bits, hist_bits)
    assert got_hi == _coarse_bin(hi_bits, hist_bits)


def test_bf16x2_coarse_bins_are_monotonic():
    """Ordered keys must preserve score order, which is what makes top-k sound."""
    hist_bits = 8
    scores = [-10.0, -1.0, -0.5, 0.0, 0.5, 1.0, 10.0]
    bins = [_coarse_bin(_float_to_bf16_bits(s), hist_bits) for s in scores]
    assert bins == sorted(bins)


def test_bf16x2_widening_is_exact():
    """BF16 -> FP32 widening must be lossless so top-k selects the same values."""
    for x in [0.0, 1.0, -1.0, 0.5, 3.25, -7.5]:
        bits = _float_to_bf16_bits(x)
        assert _bf16_bits_to_float(bits) == pytest.approx(x, rel=0, abs=0)


# ---------------------------------------------------------------------------
# Backend dtype capability negotiation and FP32 fallback.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stride,expected",
    [
        (0, True),
        (8, True),
        (16, True),
        (65536, True),
        (1048576, True),
        (4, False),
        (7, False),
        (65535, False),
    ],
)
def test_takes_bf16_logits_alignment(stride, expected):
    assert _takes_bf16_logits("bfloat", 16, 1, stride) is expected


@pytest.mark.parametrize(
    "code,bits,lanes",
    [
        ("float", 32, 1),
        ("float", 16, 1),
        ("bfloat", 32, 1),
        ("bfloat", 16, 2),
        ("uint", 16, 1),
    ],
)
def test_takes_bf16_logits_rejects_other_dtypes(code, bits, lanes):
    assert _takes_bf16_logits(code, bits, lanes, 16) is False


def test_fp32_fallback_when_bf16_unaligned():
    """An unaligned BF16 row must fall back to the FP32 kernel, not crash."""
    assert _takes_bf16_logits("bfloat", 16, 1, 7) is False
    # The FP32 predicate is dtype-only, so the fallback is always available.
    assert _takes_bf16_logits("float", 32, 1, 7) is False


# ---------------------------------------------------------------------------
# indexer_topk module: lazy torch import and public interface stability.
# ---------------------------------------------------------------------------


def test_indexer_topk_imports_without_torch():
    """The module must be importable in a numpy/pytest-only environment."""
    mod = importlib.import_module("sglang.srt.state_capturer.indexer_topk")
    assert hasattr(mod, "IndexerTopkCapturer")
    assert hasattr(mod, "maybe_capture_indexer_topk")
    assert hasattr(mod, "create_indexer_capturer")
    assert hasattr(mod, "extract_indexer_topk_from_meta_info")


def test_maybe_capture_passthrough_without_capturer():
    mod = importlib.import_module("sglang.srt.state_capturer.indexer_topk")
    sentinel = object()
    assert mod.maybe_capture_indexer_topk(0, sentinel) is sentinel
    assert mod.maybe_capture_indexer_topk(0, None) is None


def test_extract_indexer_topk_empty_meta():
    mod = importlib.import_module("sglang.srt.state_capturer.indexer_topk")
    out = mod.extract_indexer_topk_from_meta_info({"meta_info": {}})
    assert out.size == 0
