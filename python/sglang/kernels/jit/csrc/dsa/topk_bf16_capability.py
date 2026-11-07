"""Pure-Python BF16 top-k capability negotiation helpers.

This module is intentionally free of torch/CUDA/sglang imports so that the
backend dtype negotiation and the BF16x2 packed histogram extraction can be
unit-tested without a GPU or the full runtime dependency set.  The device-side
mirrors live in ``kpool_topk_transform.cuh`` (``convert_bf16x2_to_uint8x2``)
and ``topk_transform.cuh`` (``takes_bf16_logits``).
"""

from __future__ import annotations

from dataclasses import dataclass

# DLPack dtype codes, kept as plain ints so importing dlpack is not required.
DL_FLOAT = 2
DL_BFLOAT = 4


@dataclass(frozen=True)
class DType:
    code: int
    bits: int
    lanes: int = 1


FP32 = DType(DL_FLOAT, 32, 1)
BF16 = DType(DL_BFLOAT, 16, 1)


def vec_width(element_bytes: int) -> int:
    """Elements per 16-byte vector: rows and slices are aligned to this."""
    if element_bytes <= 0 or 16 % element_bytes != 0:
        raise ValueError(f"unsupported element width: {element_bytes}")
    return 16 // element_bytes


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
    """Pick the kernel dtype, falling back to FP32 after an upcast.

    Returns ``"bf16"`` when the native BF16 path is usable and ``"fp32"``
    otherwise.  Raises ``ValueError`` for dtypes the kernels cannot read at all.
    """
    if takes_bf16_logits(dtype, row_stride):
        return "bf16"
    if takes_fp32_logits(dtype):
        return "fp32"
    if dtype.code == DL_BFLOAT and dtype.bits == 16 and dtype.lanes == 1:
        # Misaligned BF16 rows: the caller upcasts and takes the FP32 kernel.
        return "fp32"
    raise ValueError(f"unsupported logits dtype: {dtype}")


def pack_bf16x2(lo: int, hi: int) -> int:
    """Pack two 16-bit BF16 bit patterns into one 32-bit word (low half first)."""
    return (lo & 0xFFFF) | ((hi & 0xFFFF) << 16)


def host_convert_bf16_to_uint8(bits: int) -> int:
    """Ordered 16-bit key of a BF16 score, matching the device-side key."""
    bits &= 0xFFFF
    key = (~bits) & 0xFFFF if (bits & 0x8000) else (bits | 0x8000)
    return (key >> 8) & 0xFF


def host_convert_bf16x2_to_uint8x2(word: int) -> tuple[int, int]:
    """Host mirror of ``convert_bf16x2_to_uint8x2``: (low half, high half)."""
    lo = word & 0xFFFF
    hi = (word >> 16) & 0xFFFF
    return host_convert_bf16_to_uint8(lo), host_convert_bf16_to_uint8(hi)
