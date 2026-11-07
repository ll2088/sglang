"""Pure-Python mirror of the DSA top-k BF16 capability negotiation.

This module intentionally has no torch / CUDA dependency so the negotiation and
fallback logic can be unit-tested in a minimal numpy/pytest environment.  The
kernel-side predicates in ``kpool_topk_transform.cuh`` and
``topk_transform.cuh`` must stay in sync with the constants here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Elements per 16-byte vector, matching kVecWidth<InT> in the CUDA kernels.
VEC_WIDTH_BF16 = 8
VEC_WIDTH_FP32 = 4

# Ordered 16-bit key of a BF16 score, matching convert_bf16_to_uint8.
def bf16_bits_to_key(bits: int) -> int:
    bits &= 0xFFFF
    if bits & 0x8000:
        return (~bits) & 0xFFFF
    return (bits | 0x8000) & 0xFFFF


def bf16_bits_to_coarse_bin(bits: int) -> int:
    """Coarse bin (top 8 bits of the ordered key) of one BF16 value."""
    return bf16_bits_to_key(bits) >> 8


def bf16x2_to_coarse_bins(word: int) -> tuple[int, int]:
    """Coarse bins of the two BF16 scores packed in ``word`` (low half first).

    Mirrors convert_bf16x2_to_uint8x2: one packed conversion and key flip.
    """
    word &= 0xFFFFFFFF
    lo = word & 0xFFFF
    hi = (word >> 16) & 0xFFFF
    return bf16_bits_to_coarse_bin(lo), bf16_bits_to_coarse_bin(hi)


@dataclass(frozen=True)
class DTypeInfo:
    code: str  # "float" or "bfloat"
    bits: int
    lanes: int = 1


@dataclass(frozen=True)
class BackendCapability:
    name: str
    takes_bf16: bool
    takes_fp32: bool


# Backends that can read BF16 logits natively.  ``cooperative`` and
# ``persistent`` are the CUDA paths; ``gfx950`` is the ROCm path.
BACKENDS: dict[str, BackendCapability] = {
    "cooperative": BackendCapability("cooperative", takes_bf16=True, takes_fp32=True),
    "persistent": BackendCapability("persistent", takes_bf16=True, takes_fp32=True),
    "gfx950": BackendCapability("gfx950", takes_bf16=True, takes_fp32=True),
    "deep_select": BackendCapability("deep_select", takes_bf16=False, takes_fp32=True),
}


def takes_bf16_logits(dtype: DTypeInfo, row_stride: int) -> bool:
    """Whether a BF16 logits tensor can be read natively.

    Rows must stay 16-byte aligned, i.e. the row stride must be a multiple of
    the BF16 vector width.
    """
    return (
        dtype.code == "bfloat"
        and dtype.bits == 16
        and dtype.lanes == 1
        and row_stride % VEC_WIDTH_BF16 == 0
    )


def takes_fp32_logits(dtype: DTypeInfo) -> bool:
    return dtype.code == "float" and dtype.bits == 32 and dtype.lanes == 1


def resolve_backend(
    requested: str,
    dtype: DTypeInfo,
    row_stride: int,
    num_rows: int,
) -> tuple[str, bool]:
    """Resolve the top-k backend and whether BF16 logits are read natively.

    Returns ``(backend_name, use_bf16)``.  ``auto`` picks ``cooperative`` for
    up to 64 rows and ``persistent`` above that, matching the scheduler.  When
    the chosen backend cannot take BF16 (or the row stride is not aligned) the
    caller must upcast to FP32 and ``use_bf16`` is False.
    """
    if requested == "auto":
        backend = "cooperative" if num_rows <= 64 else "persistent"
    else:
        backend = requested

    cap = BACKENDS.get(backend)
    if cap is None:
        raise ValueError(f"unknown top-k backend: {backend}")

    if not takes_fp32_logits(dtype) and not takes_bf16_logits(dtype, row_stride):
        raise ValueError(
            f"unsupported logits dtype for top-k: {dtype} (stride={row_stride})"
        )

    use_bf16 = (
        cap.takes_bf16
        and takes_bf16_logits(dtype, row_stride)
    )
    return backend, use_bf16


def select_logits_dtype(
    requested: str,
    dtype: DTypeInfo,
    row_stride: int,
    num_rows: int,
) -> tuple[str, DTypeInfo]:
    """Return the backend and the dtype the kernel will actually read.

    Falls back to FP32 when BF16 cannot be read natively, so callers can size
    their upcast buffer accordingly.
    """
    backend, use_bf16 = resolve_backend(requested, dtype, row_stride, num_rows)
    if use_bf16:
        return backend, dtype
    return backend, DTypeInfo(code="float", bits=32, lanes=1)
