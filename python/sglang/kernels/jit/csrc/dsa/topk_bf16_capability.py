"""Pure-Python mirrors of the DSA top-k BF16 capability negotiation.

These helpers are intentionally free of torch/CUDA/sglang imports so the
capability predicates and the packed BF16x2 histogram extraction can be
unit-tested in a minimal pytest-only job.  The CUDA kernels in
``kpool_topk_transform.cuh`` and ``dsa_gfx950/topk_transform.cuh`` keep the
same predicates; keep the two in sync.
"""

from __future__ import annotations

from typing import NamedTuple


# DLDataType codes, mirrored from dlpack so this module needs no dlpack import.
DL_FLOAT = 2
DL_BFLOAT = 4


class DType(NamedTuple):
    code: int
    bits: int
    lanes: int = 1


FP32 = DType(DL_FLOAT, 32, 1)
BF16 = DType(DL_BFLOAT, 16, 1)


def vec_width(elem_bytes: int) -> int:
    """Elements per 16-byte TMA/vector unit for an element width."""
    if elem_bytes <= 0 or 16 % elem_bytes != 0:
        raise ValueError(f"unsupported element width: {elem_bytes}")
    return 16 // elem_bytes


def takes_bf16_logits(dtype: DType, row_stride: int) -> bool:
    """BF16 logits are read natively when every row stays 16-byte aligned."""
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
    """Return ``"bf16"`` or ``"fp32"`` for the kernel to launch.

    BF16 is only taken when the row stride keeps rows 16-byte aligned;
    otherwise the caller upcasts and takes the FP32 kernel, exactly as the
    CUDA dispatcher does.
    """
    if takes_bf16_logits(dtype, row_stride):
        return "bf16"
    if takes_fp32_logits(dtype):
        return "fp32"
    raise ValueError(f"unsupported logits dtype: {dtype}")


def host_convert_bf16_to_uint8(bits: int) -> int:
    """Ordered 8-bit key of a BF16 score, matching convert_bf16_to_uint8."""
    bits &= 0xFFFF
    key = (~bits) & 0xFFFF if (bits & 0x8000) else (bits | 0x8000)
    return (key >> 8) & 0xFF


def host_convert_bf16x2_to_uint8x2(word: int) -> tuple[int, int]:
    """Coarse bins of the two BF16 scores packed in ``word`` (low half first).

    Mirrors ``convert_bf16x2_to_uint8x2``: one packed conversion and key flip
    for both halves, so the histogram pass pays no per-element conversion.
    """
    word &= 0xFFFFFFFF
    lo = word & 0xFFFF
    hi = (word >> 16) & 0xFFFF
    return host_convert_bf16_to_uint8(lo), host_convert_bf16_to_uint8(hi)


def pack_bf16x2(lo_bits: int, hi_bits: int) -> int:
    """Pack two BF16 bit patterns into one 32-bit word (low half first)."""
    return ((hi_bits & 0xFFFF) << 16) | (lo_bits & 0xFFFF)
