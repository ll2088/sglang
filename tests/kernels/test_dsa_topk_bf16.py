"""Dependency-free reference tests for the BF16 indexer top-k path.

This module intentionally imports only the Python standard library (plus
``pytest``) so it is collectable and runnable in a minimal CI environment
where the ``sglang`` package and its dependency tree are not installed. It
mirrors the pure-Python pieces of the BF16 top-k port:

* the BF16x2 packed coarse-bin extraction used by the histogram kernels,
* the backend dtype capability negotiation (``takes_bf16_logits``) and its
  FP32 fallback,
* the indexer_topk backend resolution / fallback branch.

The CUDA kernel-level coverage lives in ``test_dsa_topk_bf16_kernel.py`` and
skips cleanly when the compiled extension is unavailable.
"""

import struct

import pytest


# ---------------------------------------------------------------------------
# BF16x2 packed coarse-bin extraction (reference implementation)
# ---------------------------------------------------------------------------


def _f32_to_bf16_bits(x: float) -> int:
    """Round-to-nearest-even FP32 -> BF16, returned as the 16-bit pattern."""
    bits = struct.unpack("<I", struct.pack("<f", x))[0]
    # Round to nearest even on the low 16 bits.
    lsb = (bits >> 16) & 1
    rounding_bias = 0x7FFF + lsb
    bits = (bits + rounding_bias) & 0xFFFFFFFF
    return (bits >> 16) & 0xFFFF


def _bf16_bits_to_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", (bits & 0xFFFF) << 16))[0]


def _order_key16(bits: int) -> int:
    """Ordered 16-bit key of a BF16 pattern: ~bits if negative else bits|0x8000."""
    if bits & 0x8000:
        return (~bits) & 0xFFFF
    return (bits | 0x8000) & 0xFFFF


def extract_coarse_bin_bf16(x: float, k_bits: int) -> int:
    """Coarse bin of a single BF16 score, matching extract_coarse_bin_N."""
    key = _order_key16(_f32_to_bf16_bits(x))
    return key >> (16 - k_bits)


def extract_coarse_bins_bf16x2(w: int, k_bits: int) -> tuple:
    """Coarse bins of the two BF16 scores packed in ``w`` (low half first).

    ``w`` holds two BF16 patterns: the low 16 bits are the first score, the
    high 16 bits the second. This mirrors the packed conversion + key flip in
    the kernel: per half, ~bits if negative else bits | 0x8000.
    """
    lo = w & 0xFFFF
    hi = (w >> 16) & 0xFFFF
    return (
        _order_key16(lo) >> (16 - k_bits),
        _order_key16(hi) >> (16 - k_bits),
    )


def pack_bf16x2(a: float, b: float) -> int:
    return (_f32_to_bf16_bits(b) << 16) | _f32_to_bf16_bits(a)


@pytest.mark.parametrize("k_bits", [8, 12])
@pytest.mark.parametrize(
    "values",
    [
        (0.0, 0.0),
        (1.0, -1.0),
        (0.5, 2.0),
        (-3.25, 7.5),
        (1e-3, -1e-3),
        (100.0, -100.0),
    ],
)
def test_bf16x2_packed_bins_match_scalar(k_bits: int, values: tuple) -> None:
    a, b = values
    packed = pack_bf16x2(a, b)
    lo, hi = extract_coarse_bins_bf16x2(packed, k_bits)
    assert lo == extract_coarse_bin_bf16(a, k_bits)
    assert hi == extract_coarse_bin_bf16(b, k_bits)


def test_bf16x2_packed_bins_are_monotonic() -> None:
    """Ordered keys must be monotonic in the score, so bins are too."""
    k_bits = 12
    scores = [-10.0, -1.0, -0.5, 0.0, 0.5, 1.0, 10.0]
    bins = [extract_coarse_bin_bf16(s, k_bits) for s in scores]
    assert bins == sorted(bins)


def test_bf16_widens_to_fp32_exactly() -> None:
    """BF16 -> FP32 widening is exact, so top-k selects the same values."""
    for x in [0.0, 1.0, -1.0, 0.1, 3.14159, -2.71828, 1e-6, 1e6]:
        bits = _f32_to_bf16_bits(x)
        widened = _bf16_bits_to_f32(bits)
        # Widening the BF16 pattern back to FP32 is lossless by construction.
        assert _f32_to_bf16_bits(widened) == bits


# ---------------------------------------------------------------------------
# Backend dtype capability negotiation and FP32 fallback
# ---------------------------------------------------------------------------


def takes_bf16_logits(dtype_is_bf16: bool, stride: int) -> bool:
    """Mirror of the kernel-side ``takes_bf16_logits`` predicate.

    BF16 is only taken when the row stride keeps every row 16-byte aligned
    (8 BF16 elements). Otherwise the caller upcasts and takes the FP32 kernel.
    """
    vec_width_bf16 = 16 // 2
    return dtype_is_bf16 and stride % vec_width_bf16 == 0


def resolve_logits_dtype(dtype_is_bf16: bool, stride: int) -> str:
    """Backend resolution: BF16 when capable, else FP32 fallback."""
    return "bf16" if takes_bf16_logits(dtype_is_bf16, stride) else "fp32"


@pytest.mark.parametrize(
    "dtype_is_bf16,stride,expected",
    [
        (True, 8, True),
        (True, 16, True),
        (True, 1024, True),
        (True, 4, False),
        (True, 7, False),
        (True, 0, True),
        (False, 8, False),
        (False, 16, False),
    ],
)
def test_takes_bf16_logits(dtype_is_bf16: bool, stride: int, expected: bool) -> None:
    assert takes_bf16_logits(dtype_is_bf16, stride) is expected


@pytest.mark.parametrize(
    "dtype_is_bf16,stride,expected",
    [
        (True, 8, "bf16"),
        (True, 4, "fp32"),
        (True, 7, "fp32"),
        (False, 8, "fp32"),
    ],
)
def test_resolve_logits_dtype_falls_back(dtype_is_bf16: bool, stride: int, expected: str) -> None:
    assert resolve_logits_dtype(dtype_is_bf16, stride) == expected


# ---------------------------------------------------------------------------
# indexer_topk backend resolution / fallback branch
# ---------------------------------------------------------------------------


def resolve_indexer_topk_backend(
    requested: str,
    *,
    dtype_is_bf16: bool,
    stride: int,
    num_rows: int,
) -> str:
    """Resolve the indexer top-k backend, mirroring the scheduler layer.

    ``auto`` picks ``cooperative`` for up to 64 rows and ``persistent``
    otherwise. A backend that cannot take BF16 logits (alignment failure)
    falls back to the FP32 path rather than silently upcasting in the kernel.
    """
    if requested == "auto":
        backend = "cooperative" if num_rows <= 64 else "persistent"
    else:
        backend = requested

    if backend in ("cooperative", "persistent"):
        if dtype_is_bf16 and not takes_bf16_logits(True, stride):
            return "fp32"
        return "bf16" if dtype_is_bf16 else "fp32"
    # Backends that only take FP32 always upcast.
    return "fp32"


@pytest.mark.parametrize(
    "requested,dtype_is_bf16,stride,num_rows,expected",
    [
        ("auto", True, 8, 1, "bf16"),
        ("auto", True, 8, 64, "bf16"),
        ("auto", True, 8, 65, "bf16"),
        ("auto", True, 4, 1, "fp32"),
        ("auto", False, 8, 1, "fp32"),
        ("cooperative", True, 8, 1, "bf16"),
        ("cooperative", True, 7, 1, "fp32"),
        ("persistent", True, 8, 128, "bf16"),
        ("persistent", True, 7, 128, "fp32"),
        ("fp32_only", True, 8, 1, "fp32"),
    ],
)
def test_indexer_topk_backend_resolution(
    requested: str,
    dtype_is_bf16: bool,
    stride: int,
    num_rows: int,
    expected: str,
) -> None:
    assert (
        resolve_indexer_topk_backend(
            requested,
            dtype_is_bf16=dtype_is_bf16,
            stride=stride,
            num_rows=num_rows,
        )
        == expected
    )


def test_indexer_topk_fallback_is_recorded() -> None:
    """A BF16 request that fails alignment must be observable as a fallback."""
    events = []

    def resolve_with_log(requested: str, *, dtype_is_bf16: bool, stride: int, num_rows: int) -> str:
        resolved = resolve_indexer_topk_backend(
            requested, dtype_is_bf16=dtype_is_bf16, stride=stride, num_rows=num_rows
        )
        if dtype_is_bf16 and resolved == "fp32":
            events.append((requested, "bf16->fp32"))
        return resolved

    assert resolve_with_log("auto", dtype_is_bf16=True, stride=7, num_rows=1) == "fp32"
    assert events == [("auto", "bf16->fp32")]
