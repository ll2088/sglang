"""Pure-Python BF16 capability negotiation for the DSA indexer top-k.

This module is intentionally free of torch / sglang runtime imports so the
dtype/alignment negotiation and the host-side mirror of the BF16x2 packed
histogram extraction can be unit-tested with only numpy + pytest.

The CUDA/HIP kernels in this directory mirror these predicates; keep the two
in sync when changing the alignment rule.
"""

from __future__ import annotations

# DLDataType codes we care about (dlpack).
kDLFloat = 2
kDLBfloat = 4

FP32 = "fp32"
BF16 = "bf16"

# Element widths in bytes.
_WIDTH = {FP32: 4, BF16: 2}


def vec_width(elem_bytes: int) -> int:
    """Elements per 16-byte vector unit for an element width in bytes."""
    if elem_bytes <= 0 or 16 % elem_bytes != 0:
        raise ValueError(f"element width {elem_bytes} does not divide 16")
    return 16 // elem_bytes


def tma_align(elem_bytes: int) -> int:
    """Row-stride granularity that keeps every row 16-byte aligned."""
    return vec_width(elem_bytes)


def takes_fp32_logits(dtype_code: int, dtype_bits: int, dtype_lanes: int) -> bool:
    return dtype_code == kDLFloat and dtype_bits == 32 and dtype_lanes == 1


def takes_bf16_logits(dtype_code: int, dtype_bits: int, dtype_lanes: int, row_stride: int) -> bool:
    """BF16 logits are read natively only when every row stays 16-byte aligned."""
    if dtype_code != kDLBfloat or dtype_bits != 16 or dtype_lanes != 1:
        return False
    if row_stride <= 0:
        return False
    return row_stride % tma_align(_WIDTH[BF16]) == 0


def resolve_logits_dtype(dtype_code: int, dtype_bits: int, dtype_lanes: int, row_stride: int) -> str:
    """Pick the kernel dtype, falling back to FP32 when BF16 is not usable."""
    if takes_bf16_logits(dtype_code, dtype_bits, dtype_lanes, row_stride):
        return BF16
    return FP32


# ---------------------------------------------------------------------------
# Host-side mirror of convert_bf16x2_to_uint8x2 (see kpool_topk_transform.cuh).
# ---------------------------------------------------------------------------


def host_convert_bf16_to_uint8(bits: int) -> int:
    bits &= 0xFFFF
    key = (~bits) & 0xFFFF if (bits & 0x8000) else (bits | 0x8000)
    return (key >> 8) & 0xFF


def host_convert_bf16x2_to_uint8x2(w: int) -> tuple[int, int]:
    lo = w & 0xFFFF
    hi = (w >> 16) & 0xFFFF
    return host_convert_bf16_to_uint8(lo), host_convert_bf16_to_uint8(hi)
