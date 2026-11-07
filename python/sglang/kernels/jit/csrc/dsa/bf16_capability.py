"""Pure-Python BF16 logits capability negotiation for the DSA indexer top-k.

This module is intentionally free of torch, sglang runtime and CUDA imports so
that the dtype/alignment negotiation can be unit-tested in a minimal CI job.
It mirrors the host-side predicates in ``kpool_topk_transform.cuh`` and
``topk_transform.cuh``: BF16 logits are read natively only when the row stride
keeps every row 16-byte aligned, otherwise the caller upcasts to FP32.
"""

from __future__ import annotations

# DLDataType codes used by the negotiation.  Kept as plain ints so this module
# does not need dlpack/tvm-ffi at import time.
BF16 = "bfloat16"
FP32 = "float32"

# Bytes per TMA/vector unit.  The alignment granularity is derived from the
# element width, not hardcoded to the FP32 case.
_VEC_BYTES = 16


def vec_width(elem_bytes: int) -> int:
    """Elements per 16-byte vector unit for an element of ``elem_bytes``."""
    if elem_bytes <= 0 or _VEC_BYTES % elem_bytes != 0:
        raise ValueError(f"unsupported element width: {elem_bytes}")
    return _VEC_BYTES // elem_bytes


def tma_align(elem_bytes: int) -> int:
    """Row-stride multiple required for 16-byte aligned TMA/vector loads."""
    return vec_width(elem_bytes)


def takes_bf16_logits(dtype: str, row_stride: int) -> bool:
    """Whether BF16 logits can be read natively for this dtype/stride."""
    if dtype != BF16:
        return False
    if row_stride <= 0:
        return False
    return row_stride % tma_align(2) == 0


def takes_fp32_logits(dtype: str) -> bool:
    """FP32 logits are always readable; named for dispatcher symmetry."""
    return dtype == FP32


def resolve_logits_dtype(dtype: str, row_stride: int) -> str:
    """Resolve the dtype the top-k kernel should read.

    BF16 is taken natively when aligned; otherwise the caller upcasts and the
    FP32 kernel is used.  FP32 is always taken as-is.
    """
    if takes_bf16_logits(dtype, row_stride):
        return BF16
    return FP32
