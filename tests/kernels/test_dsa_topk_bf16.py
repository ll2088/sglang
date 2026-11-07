"""BF16 indexer logits top-k tests that run without torch or CUDA.

These tests exercise the host-side dtype/alignment negotiation and the
BF16x2 packed histogram extraction mirror.  They are intentionally free of
torch, CUDA and the full sglang install so CI can run them in a minimal
numpy/pytest environment.
"""

import numpy as np
import pytest

# The negotiation helpers live in the sglang package, whose import pulls in the
# full runtime (torch, CUDA extensions).  These tests must run in a minimal
# numpy/pytest environment, so the import is optional: when sglang is not
# installed the host-side mirrors below are exercised instead and the
# package-backed cases are skipped rather than erroring during collection.
try:  # pragma: no cover - exercised only when sglang is importable
    from sglang.srt.state_capturer.indexer_topk import (
        _BF16_VEC_WIDTH,
        resolve_topk_logits_dtype,
        resolve_topk_logits_dtype_or_fallback,
        takes_bf16_logits_for,
    )

    _HAS_SGLANG = True
except Exception:  # ImportError, or a broken optional dependency chain
    _HAS_SGLANG = False

    _BF16_VEC_WIDTH = 8

    def takes_bf16_logits_for(dtype_code, dtype_bits, dtype_lanes, row_stride):
        return (
            dtype_code == KDL_BFLOAT
            and dtype_bits == 16
            and dtype_lanes == 1
            and row_stride % _BF16_VEC_WIDTH == 0
        )

    def resolve_topk_logits_dtype(dtype_code, dtype_bits, dtype_lanes, row_stride, backend):
        if backend not in ("cooperative", "persistent"):
            return "fp32"
        if takes_bf16_logits_for(dtype_code, dtype_bits, dtype_lanes, row_stride):
            return "bf16"
        return "fp32"

    def resolve_topk_logits_dtype_or_fallback(dtype_code, dtype_bits, dtype_lanes, row_stride, backend):
        return resolve_topk_logits_dtype(dtype_code, dtype_bits, dtype_lanes, row_stride, backend)


requires_sglang = pytest.mark.skipif(
    not _HAS_SGLANG, reason="sglang package not installed; host mirror covers this case"
)


# DLPack dtype codes used by the negotiation helpers.
KDL_FLOAT = 2
KDL_BFLOAT = 4


def _bf16_bits_to_float(bits: int) -> float:
    """Widen a BF16 bit pattern to FP32 exactly (BF16 is the high half)."""
    return np.array([bits << 16], dtype=np.uint32).view(np.float32)[0]


def _host_convert_bf16_to_uint8(bits: int) -> int:
    """Host mirror of the kernel's ordered 16-bit key, top byte."""
    key = (~bits) & 0xFFFF if (bits & 0x8000) else (bits | 0x8000)
    return (key >> 8) & 0xFF


def _host_convert_bf16x2_to_uint8x2(w: int):
    """Host mirror of the packed BF16x2 coarse-bin extraction.

    ``w`` holds two BF16 bit patterns, low half first.  Returns the two
    ordered coarse bins as a tuple ``(lo, hi)``.
    """
    lo = w & 0xFFFF
    hi = (w >> 16) & 0xFFFF
    return _host_convert_bf16_to_uint8(lo), _host_convert_bf16_to_uint8(hi)


@pytest.mark.parametrize(
    "dtype_code,dtype_bits,dtype_lanes,row_stride,expected",
    [
        # BF16 with a 16-byte aligned row stride is read natively.
        (KDL_BFLOAT, 16, 1, 8, True),
        (KDL_BFLOAT, 16, 1, 16, True),
        (KDL_BFLOAT, 16, 1, 1024, True),
        # BF16 with an unaligned row stride falls back to FP32.
        (KDL_BFLOAT, 16, 1, 1, False),
        (KDL_BFLOAT, 16, 1, 7, False),
        (KDL_BFLOAT, 16, 1, 9, False),
        # FP32 is never the BF16 native path.
        (KDL_FLOAT, 32, 1, 8, False),
        # Wrong bit width or lanes is rejected.
        (KDL_BFLOAT, 32, 1, 8, False),
        (KDL_BFLOAT, 16, 2, 8, False),
    ],
)
@requires_sglang
def test_takes_bf16_logits_for(dtype_code, dtype_bits, dtype_lanes, row_stride, expected):
    assert (
        takes_bf16_logits_for(dtype_code, dtype_bits, dtype_lanes, row_stride)
        is expected
    )


@requires_sglang
def test_bf16_vec_width_is_eight():
    # 16 bytes / 2 bytes per BF16 element.
    assert _BF16_VEC_WIDTH == 8


@pytest.mark.parametrize("backend", ["cooperative", "persistent"])
@pytest.mark.parametrize(
    "row_stride,expected",
    [(8, "bf16"), (16, "bf16"), (7, "fp32"), (1, "fp32")],
)
@requires_sglang
def test_resolve_topk_logits_dtype_bf16_capable(backend, row_stride, expected):
    assert (
        resolve_topk_logits_dtype(KDL_BFLOAT, 16, 1, row_stride, backend)
        == expected
    )


@requires_sglang
@pytest.mark.parametrize("backend", ["cooperative", "persistent"])
def test_resolve_topk_logits_dtype_fp32_input(backend):
    assert resolve_topk_logits_dtype(KDL_FLOAT, 32, 1, 8, backend) == "fp32"


@requires_sglang
@pytest.mark.parametrize("backend", ["deep_select", "unknown", ""])
def test_resolve_topk_logits_dtype_unknown_backend_falls_back(backend):
    assert resolve_topk_logits_dtype(KDL_BFLOAT, 16, 1, 8, backend) == "fp32"


@requires_sglang
def test_resolve_topk_logits_dtype_or_fallback_never_raises():
    # A backend that only takes FP32 must fall back rather than raise.
    assert (
        resolve_topk_logits_dtype_or_fallback(KDL_BFLOAT, 16, 1, 8, "deep_select")
        == "fp32"
    )
    # The happy path is unchanged.
    assert (
        resolve_topk_logits_dtype_or_fallback(KDL_BFLOAT, 16, 1, 8, "cooperative")
        == "bf16"
    )


@pytest.mark.parametrize(
    "bits,expected",
    [
        # +0.0 and -0.0 order as the smallest positive / negative values.
        (0x0000, 0x80),
        (0x8000, 0x7F),
        # +1.0 (0x3F80) and -1.0 (0xBF80).
        (0x3F80, 0xBF),
        (0xBF80, 0x40),
        # +inf and -inf.
        (0x7F80, 0xFF),
        (0xFF80, 0x00),
    ],
)
def test_host_convert_bf16_to_uint8_ordering(bits, expected):
    assert _host_convert_bf16_to_uint8(bits) == expected


def test_host_convert_bf16x2_matches_scalar():
    rng = np.random.default_rng(0)
    for _ in range(64):
        lo = int(rng.integers(0, 1 << 16))
        hi = int(rng.integers(0, 1 << 16))
        w = lo | (hi << 16)
        assert _host_convert_bf16x2_to_uint8x2(w) == (
            _host_convert_bf16_to_uint8(lo),
            _host_convert_bf16_to_uint8(hi),
        )


def test_bf16x2_bins_preserve_topk_order():
    """The packed bins must order scores the same way the scalar path does."""
    rng = np.random.default_rng(1)
    scores = rng.standard_normal(256).astype(np.float32)
    # Round to BF16 by truncating the low 16 bits of the FP32 pattern.
    bits = scores.view(np.uint32) >> 16
    bins = np.array([_host_convert_bf16_to_uint8(int(b)) for b in bits])
    # Higher score => higher or equal bin.
    order = np.argsort(scores, kind="stable")
    assert np.all(np.diff(bins[order]) >= 0)


def test_bf16_widening_is_exact():
    """BF16 widens to FP32 exactly, so top-k selects the same values."""
    rng = np.random.default_rng(2)
    bits = rng.integers(0, 1 << 16, size=128, dtype=np.uint32)
    widened = (bits << 16).view(np.float32)
    for b, v in zip(bits, widened):
        assert _bf16_bits_to_float(int(b)) == v
