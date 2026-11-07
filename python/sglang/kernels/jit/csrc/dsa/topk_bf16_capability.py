"""Pure-Python BF16 indexer top-k capability negotiation.

This module is deliberately free of torch, CUDA and sglang imports so the
negotiation logic and the BF16x2 packed histogram extraction can be unit
tested in a minimal pytest-only environment.  The CUDA kernels mirror these
predicates (see ``takes_bf16_logits`` in the DSA top-k sources); keeping a
host-side copy here lets CI verify the dtype/alignment contract without a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

# DLPack dtype codes, duplicated here so importing this module never pulls in
# dlpack/torch.  Values match ``DLDataTypeCode``.
DL_FLOAT = 2
DL_BFLOAT = 4


@dataclass(frozen=True)
class DType:
    """Minimal stand-in for ``DLDataType`` (code, bits, lanes)."""

    code: int
    bits: int
    lanes: int = 1


FP32 = DType(DL_FLOAT, 32, 1)
BF16 = DType(DL_BFLOAT, 16, 1)

# 16-byte TMA/vector unit: rows and slices are aligned to this many elements.
VEC_BYTES = 16


def vec_width(element_bytes: int) -> int:
    """Elements per 16-byte vector for a given element width.

    The alignment granularity is derived from the element width instead of
    being hardcoded to the FP32 value of 4.
    """
    if element_bytes <= 0 or VEC_BYTES % element_bytes != 0:
        raise ValueError(f"unsupported element width: {element_bytes} bytes")
    return VEC_BYTES // element_bytes


def takes_bf16_logits(dtype: DType, row_stride: int) -> bool:
    """Whether the top-k kernel can read BF16 logits natively.

    Requires a scalar BF16 dtype and a row stride that keeps every row
    16-byte aligned; otherwise the caller upcasts and takes the FP32 kernel.
    """
    return (
        dtype.code == DL_BFLOAT
        and dtype.bits == 16
        and dtype.lanes == 1
        and row_stride % vec_width(2) == 0
    )


def takes_fp32_logits(dtype: DType) -> bool:
    """FP32 logits are always readable; named so the dispatcher can negotiate."""
    return dtype.code == DL_FLOAT and dtype.bits == 32 and dtype.lanes == 1


def resolve_logits_dtype(dtype: DType, row_stride: int) -> str:
    """Pick the kernel dtype for the given logits, falling back to FP32.

    Returns ``"bf16"`` when the native BF16 path applies and ``"fp32"``
    otherwise (including misaligned BF16 rows, which the caller upcasts).
    """
    if takes_bf16_logits(dtype, row_stride):
        return "bf16"
    if takes_fp32_logits(dtype):
        return "fp32"
    if dtype.code == DL_BFLOAT and dtype.bits == 16 and dtype.lanes == 1:
        # BF16 that is not 16-byte aligned: upcast and use the FP32 kernel.
        return "fp32"
    raise ValueError(f"unsupported logits dtype: {dtype}")


def pack_bf16x2(lo: int, hi: int) -> int:
    """Pack two 16-bit BF16 bit patterns into one 32-bit word (low half first)."""
    return (lo & 0xFFFF) | ((hi & 0xFFFF) << 16)


def host_convert_bf16_to_uint8(bits: int) -> int:
    """Ordered 8-bit key of a BF16 bit pattern, matching the kernel's key flip."""
    bits &= 0xFFFF
    key = (~bits) & 0xFFFF if (bits & 0x8000) else (bits | 0x8000)
    return (key >> 8) & 0xFF


def host_convert_bf16x2_to_uint8x2(word: int) -> tuple[int, int]:
    """Host mirror of the packed BF16x2 coarse-bin extraction."""
    lo = word & 0xFFFF
    hi = (word >> 16) & 0xFFFF
    return host_convert_bf16_to_uint8(lo), host_convert_bf16_to_uint8(hi)
