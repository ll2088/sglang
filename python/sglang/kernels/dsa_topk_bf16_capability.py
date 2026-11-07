"""Pure-Python DSA indexer top-k BF16 capability negotiation helpers.

This module is deliberately dependency-free (standard library only) so the
capability negotiation and the BF16x2 packed histogram extraction can be unit
tested without importing the ``sglang`` package, torch, or any CUDA runtime.

The device-side counterparts live in
``python/sglang/kernels/jit/csrc/dsa/kpool_topk_transform.cuh`` and
``python/sglang/kernels/jit/csrc/dsa_gfx950/topk_transform.cuh``; the host-side
mirrors here must stay bit-identical to them.
"""

from __future__ import annotations

from typing import Tuple

# DLDataType codes, mirrored from dlpack/dlpack.h so this module needs no
# dlpack import.
KDL_FLOAT = 2
KDL_BFLOAT = 4

# Element widths the top-k kernels are built for.
_SUPPORTED_ELEM_BYTES = (1, 2, 4, 8)


def vec_width(elem_bytes: int) -> int:
    """Elements per 16-byte TMA/vector unit for a given element width.

    Rows, per-block slices and copy sizes are aligned to this, so the
    alignment granularity follows the element width instead of being
    hardcoded to the FP32 value of 4.
    """
    if elem_bytes not in _SUPPORTED_ELEM_BYTES:
        raise ValueError(f"unsupported element width: {elem_bytes} bytes")
    return 16 // elem_bytes


def takes_bf16_logits_for(
    dtype_code: int, dtype_bits: int, dtype_lanes: int, row_stride: int
) -> bool:
    """Host-side mirror of the kernels' ``takes_bf16_logits`` predicate.

    BF16 logits are read natively only when the dtype is scalar BF16 and the
    row stride keeps every row 16-byte aligned; otherwise the caller upcasts
    and takes the FP32 kernel.
    """
    return (
        dtype_code == KDL_BFLOAT
        and dtype_bits == 16
        and dtype_lanes == 1
        and row_stride % vec_width(2) == 0
    )


def takes_fp32_logits_for(dtype_code: int, dtype_bits: int, dtype_lanes: int) -> bool:
    """FP32 logits are always readable; kept as a named capability."""
    return dtype_code == KDL_FLOAT and dtype_bits == 32 and dtype_lanes == 1


def resolve_logits_dtype(
    dtype_code: int, dtype_bits: int, dtype_lanes: int, row_stride: int
) -> str:
    """Negotiate the kernel dtype, falling back to ``"fp32"``.

    Returns ``"bf16"`` when the BF16 native-read capability holds, otherwise
    ``"fp32"``.  The fallback is explicit so the dispatcher never performs an
    implicit conversion inside the kernel.
    """
    if takes_bf16_logits_for(dtype_code, dtype_bits, dtype_lanes, row_stride):
        return "bf16"
    return "fp32"


def host_convert_bf16_to_uint8(bits: int) -> int:
    """Ordered 8-bit key of a BF16 score, matching the device helper."""
    bits &= 0xFFFF
    key = (~bits) & 0xFFFF if (bits & 0x8000) else (bits | 0x8000)
    return (key >> 8) & 0xFF


def host_convert_bf16x2_to_uint8x2(word: int) -> Tuple[int, int]:
    """Coarse bins of the two BF16 scores packed in ``word`` (low half first).

    Mirrors ``convert_bf16x2_to_uint8x2`` in the CUDA/HIP kernels: one packed
    conversion and key flip instead of two scalar conversions.
    """
    word &= 0xFFFFFFFF
    lo = word & 0xFFFF
    hi = (word >> 16) & 0xFFFF
    return host_convert_bf16_to_uint8(lo), host_convert_bf16_to_uint8(hi)


def pack_bf16x2(lo_bits: int, hi_bits: int) -> int:
    """Pack two 16-bit BF16 patterns into one 32-bit word (low half first)."""
    return ((hi_bits & 0xFFFF) << 16) | (lo_bits & 0xFFFF)
