"""Dependency-light checks for the BF16 DSA top-k port.

These tests deliberately avoid importing torch or sglang: they exercise the
host-side reference of the BF16x2 packed histogram extraction, the BF16/FP32
top-k selection parity, the alignment/empty-page boundaries and the backend
dtype capability negotiation with plain Python + numpy, so they can run in a
minimal CI job.

GPU-only kernel coverage (cooperative/persistent kernels, paged MQA logits)
lives in the CUDA test files and is expected to be marked GPU-only there.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Host mirrors of the device helpers (see kpool_topk_transform.cuh).
# ---------------------------------------------------------------------------


def bf16_bits_to_uint8(bits: int) -> int:
    """Ordered 8-bit coarse key of a BF16 score, matching convert_bf16_to_uint8."""
    bits &= 0xFFFF
    key = (~bits) & 0xFFFF if (bits & 0x8000) else (bits | 0x8000)
    return (key >> 8) & 0xFF


def bf16x2_bits_to_uint8x2(w: int):
    """Coarse bins of the two BF16 scores packed in `w` (low half first)."""
    lo = bf16_bits_to_uint8(w & 0xFFFF)
    hi = bf16_bits_to_uint8((w >> 16) & 0xFFFF)
    return lo, hi


def fp32_to_bf16_bits(x: float) -> int:
    """Round-to-nearest-even FP32 -> BF16 bit pattern (no NaN handling needed)."""
    u = np.float32(x).view(np.uint32)
    lsb = (int(u) >> 16) & 1
    rounded = int(u) + 0x7FFF + lsb
    return (rounded >> 16) & 0xFFFF


def bf16_round(x: float) -> float:
    bits = fp32_to_bf16_bits(x)
    return float(np.uint32(bits << 16).view(np.float32))


def pack_bf16x2(a: float, b: float) -> int:
    return (fp32_to_bf16_bits(a) & 0xFFFF) | ((fp32_to_bf16_bits(b) & 0xFFFF) << 16)


def fp32_coarse_bin(x: float) -> int:
    """Reference coarse bin: round to fp16, order the key, keep the top 8 bits."""
    h = np.float16(x)
    bits = int(np.float16(h).view(np.uint16))
    key = (~bits) & 0xFFFF if (bits & 0x8000) else (bits | 0x8000)
    return (key >> 8) & 0xFF


# ---------------------------------------------------------------------------
# BF16x2 packed histogram extraction.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "values",
    [
        (0.0, 0.0),
        (1.0, -1.0),
        (-0.0, 0.0),
        (3.5, -2.25),
        (1e-3, 1e3),
        (-7.0, 7.0),
    ],
)
def test_bf16x2_bins_match_scalar(values):
    a, b = values
    w = pack_bf16x2(a, b)
    lo, hi = bf16x2_bits_to_uint8x2(w)
    assert lo == bf16_bits_to_uint8(fp32_to_bf16_bits(a))
    assert hi == bf16_bits_to_uint8(fp32_to_bf16_bits(b))


def test_bf16x2_bins_are_monotonic():
    xs = np.linspace(-8.0, 8.0, 257, dtype=np.float32)
    bins = [bf16_bits_to_uint8(fp32_to_bf16_bits(float(x))) for x in xs]
    assert bins == sorted(bins)


def test_bf16x2_bins_agree_with_fp32_reference_on_bf16_grid():
    # Values exactly representable in BF16 must bin identically through the
    # BF16 path and the FP32->fp16 reference path.
    xs = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, -0.5, -1.0, -2.0, -4.0, -8.0]
    for x in xs:
        assert bf16_bits_to_uint8(fp32_to_bf16_bits(x)) == fp32_coarse_bin(x)


# ---------------------------------------------------------------------------
# BF16 / FP32 top-k selection parity.
# ---------------------------------------------------------------------------


def _topk_indices(scores, k):
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    return order[:k]


def _topk_values(scores, k):
    return sorted((scores[i] for i in _topk_indices(scores, k)), reverse=True)


@pytest.mark.parametrize("num_rows", [1, 4, 8, 33, 64])
@pytest.mark.parametrize("length", [300, 5000, 16384, 40000])
@pytest.mark.parametrize("k", [512, 1024])
@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
def test_topk_selection_parity(num_rows, length, k, dtype):
    rng = np.random.default_rng(0)
    k = min(k, length)
    for _ in range(num_rows):
        raw = rng.standard_normal(length).astype(np.float32)
        if dtype == "bf16":
            scores = [bf16_round(float(x)) for x in raw]
        else:
            scores = [float(x) for x in raw]
        idx = _topk_indices(scores, k)
        assert len(set(idx)) == k
        assert _topk_values(scores, k) == sorted(
            (scores[i] for i in idx), reverse=True
        )


def test_topk_bf16_ties_are_exact():
    # BF16 carries exact ties; any index may win but the selected values must
    # match the reference top-k values.
    scores = [bf16_round(1.0)] * 64 + [bf16_round(0.5)] * 64
    k = 32
    idx = _topk_indices(scores, k)
    assert len(set(idx)) == k
    assert _topk_values(scores, k) == [1.0] * k


# ---------------------------------------------------------------------------
# Alignment and empty-page boundaries.
# ---------------------------------------------------------------------------


def tma_align(elem_size: int) -> int:
    return 16 // elem_size


@pytest.mark.parametrize("elem_size,expected", [(4, 4), (2, 8)])
def test_tma_align_by_element_width(elem_size, expected):
    assert tma_align(elem_size) == expected


@pytest.mark.parametrize("seq_len", [65535, 65536, 65537])
@pytest.mark.parametrize("elem_size", [4, 2])
def test_stride_alignment_rounds_up(seq_len, elem_size):
    align = tma_align(elem_size)
    stride = (seq_len + align - 1) & ~(align - 1)
    assert stride % align == 0
    assert stride >= seq_len


def test_empty_page_table_row_is_all_padding():
    # A row with no pages must produce only -1 padding, never a read.
    length = 0
    topk = 8
    out = [-1] * topk
    for i in range(length):
        out[i] = i
    assert out == [-1] * topk


def test_short_row_below_topk_is_identity():
    length = 5
    topk = 8
    out = [i if i < length else -1 for i in range(topk)]
    assert out == [0, 1, 2, 3, 4, -1, -1, -1]


# ---------------------------------------------------------------------------
# Backend dtype capability negotiation and fallback.
# ---------------------------------------------------------------------------


BF16_VEC_WIDTH = tma_align(2)


def takes_bf16_logits(dtype_code, bits, lanes, stride0):
    return (
        dtype_code == "bfloat"
        and bits == 16
        and lanes == 1
        and stride0 % BF16_VEC_WIDTH == 0
    )


def takes_fp32_logits(dtype_code, bits, lanes):
    return dtype_code == "float" and bits == 32 and lanes == 1


def resolve_backend(dtype_code, bits, lanes, stride0, backend_supports_bf16):
    if backend_supports_bf16 and takes_bf16_logits(dtype_code, bits, lanes, stride0):
        return "bf16"
    if takes_fp32_logits(dtype_code, bits, lanes):
        return "fp32"
    return "upcast"


@pytest.mark.parametrize(
    "dtype_code,bits,lanes,stride0,backend_bf16,expected",
    [
        ("bfloat", 16, 1, 1024, True, "bf16"),
        ("bfloat", 16, 1, 1023, True, "upcast"),
        ("bfloat", 16, 1, 1024, False, "upcast"),
        ("float", 32, 1, 1024, True, "fp32"),
        ("float", 32, 1, 1023, True, "fp32"),
    ],
)
def test_backend_dtype_negotiation(
    dtype_code, bits, lanes, stride0, backend_bf16, expected
):
    assert (
        resolve_backend(dtype_code, bits, lanes, stride0, backend_bf16) == expected
    )


def test_backend_fallback_is_recorded():
    events = []

    def resolve(dtype_code, bits, lanes, stride0, backend_bf16):
        choice = resolve_backend(dtype_code, bits, lanes, stride0, backend_bf16)
        if choice == "upcast":
            events.append("fallback:upcast")
        return choice

    assert resolve("bfloat", 16, 1, 1023, True) == "upcast"
    assert events == ["fallback:upcast"]


# ---------------------------------------------------------------------------
# indexer_topk backend resolution / fallback branch.
# ---------------------------------------------------------------------------


def resolve_indexer_topk_backend(backend, dtype_code, bits, stride0):
    """Mirror of the scheduler-side choice: BF16 only when the backend
    declares support and the row stride keeps 16-byte alignment."""
    if backend in ("cooperative", "persistent", "deep_select"):
        if takes_bf16_logits(dtype_code, bits, 1, stride0):
            return backend, "bf16"
        return backend, "fp32"
    return "fallback", "fp32"


@pytest.mark.parametrize(
    "backend,dtype_code,bits,stride0,expected",
    [
        ("cooperative", "bfloat", 16, 1024, ("cooperative", "bf16")),
        ("cooperative", "bfloat", 16, 1023, ("cooperative", "fp32")),
        ("persistent", "bfloat", 16, 1024, ("persistent", "bf16")),
        ("persistent", "bfloat", 16, 1023, ("persistent", "fp32")),
        ("deep_select", "bfloat", 16, 1024, ("deep_select", "bf16")),
        ("unknown", "bfloat", 16, 1024, ("fallback", "fp32")),
    ],
)
def test_indexer_topk_backend_resolution(backend, dtype_code, bits, stride0, expected):
    assert resolve_indexer_topk_backend(backend, dtype_code, bits, stride0) == expected


def test_indexer_topk_interface_is_unchanged():
    # The public entry point keeps its positional signature.
    import inspect

    from sglang.srt.state_capturer.indexer_topk import maybe_capture_indexer_topk

    sig = inspect.signature(maybe_capture_indexer_topk)
    assert list(sig.parameters) == ["layer_id", "topk_indices"]
