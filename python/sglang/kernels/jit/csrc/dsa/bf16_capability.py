"""Host-side BF16 capability negotiation for the DSA indexer top-k.

This module is deliberately free of torch / CUDA / sglang runtime imports so
that the dtype/alignment negotiation can be unit-tested in a minimal pytest
job.  The predicates here mirror the device-side ones in
``kpool_topk_transform.cuh`` and ``dsa_gfx950/topk_transform.cuh``.
"""

from __future__ import annotations

from typing import NamedTuple

# DLDataType codes we care about (dlpack).
KDL_FLOAT = 2
KDL_BFLOAT = 4

# 16-byte TMA / vector unit, in bytes.
_TMA_BYTES = 16


def vec_width(element_bytes: int) -> int:
    """Elements per 16-byte vector for a given element width."""
    if element_bytes <= 0 or _TMA_BYTES % element_bytes != 0:
        raise ValueError(f"unsupported element width: {element_bytes}")
    return _TMA_BYTES // element_bytes


class DType(NamedTuple):
    code: int
    bits: int
    lanes: int = 1


FP32 = DType(KDL_FLOAT, 32, 1)
BF16 = DType(KDL_BFLOAT, 16, 1)


def takes_bf16_logits(dtype: DType, row_stride: int) -> bool:
    """Whether the top-k kernel can read BF16 logits natively.

    BF16 is read natively only when the row stride keeps every row 16-byte
aligned; otherwise the caller upcasts to FP32 and takes the FP32 kernel.
    """
    if dtype != BF16:
        return False
    if row_stride <= 0:
        return False
    return row_stride % vec_width(2) == 0


def takes_fp32_logits(dtype: DType) -> bool:
    """FP32 logits are always readable; kept as a named capability."""
    return dtype == FP32


def resolve_logits_dtype(dtype: DType, row_stride: int) -> str:
    """Negotiate the kernel dtype, falling back to FP32 when BF16 is not
    natively readable.  Returns ``"bf16"`` or ``"fp32"``."""
    if takes_bf16_logits(dtype, row_stride):
        return "bf16"
    return "fp32"


def tma_align(element_bytes: int) -> int:
    """Elements per 16-byte TMA unit for a given element width."""
    return vec_width(element_bytes)
