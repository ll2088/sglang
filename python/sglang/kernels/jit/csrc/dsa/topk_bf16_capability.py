"""Pure-Python mirror of the DSA indexer top-k BF16 capability negotiation.

This module is intentionally free of torch, CUDA and sglang imports so the
dtype/alignment negotiation and the BF16x2 packed histogram extraction can be
unit-tested on any machine.  The CUDA kernels in this directory implement the
same predicates; keep the two in sync.
"""

from __future__ import annotations

from dataclasses import dataclass

# DLPack dtype codes we care about (dlpack/dlpack.h).
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
    """BF16 logits are read natively only when every row stays 16B aligned."""
    return (
        dtype.code == DL_BFLOAT
        and dtype.bits == 16
        and dtype.lanes == 1
        and row_stride % vec_width(2) == 0
    )


def takes_fp32_logits(dtype: DType) -> bool:
    return dtype.code == DL_FLOAT and dtype.bits == 32 and dtype.lanes == 1


def resolve_logits_dtype(dtype: DType, row_stride: int) -> str:
    """Pick the kernel dtype, falling back to FP32 when BF16 is not usable.

    Returns ``"bf16"`` or ``"fp32"``.  Unsupported dtypes raise, so a caller
    cannot silently take a path the kernels do not implement.
    """
    if takes_bf16_logits(dtype, row_stride):
        return "bf16"
    if takes_fp32_logits(dtype):
        return "fp32"
    raise ValueError(f"unsupported logits dtype: {dtype}")


def host_convert_bf16_to_uint8(bits: int) -> int:
    """Ordered 8-bit key of a BF16 bit pattern (high byte of the ordered key)."""
    bits &= 0xFFFF
    key = (~bits) & 0xFFFF if (bits & 0x8000) else (bits | 0x8000)
    return (key >> 8) & 0xFF


def pack_bf16x2(lo: int, hi: int) -> int:
    return ((lo & 0xFFFF) | ((hi & 0xFFFF) << 16)) & 0xFFFFFFFF


def host_convert_bf16x2_to_uint8x2(word: int) -> tuple[int, int]:
    """Host mirror of the kernel's packed BF16x2 coarse-bin extraction."""
    word &= 0xFFFFFFFF
    lo = word & 0xFFFF
    hi = (word >> 16) & 0xFFFF
    return host_convert_bf16_to_uint8(lo), host_convert_bf16_to_uint8(hi)
