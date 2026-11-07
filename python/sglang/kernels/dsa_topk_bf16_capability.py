"""Pure-Python BF16 capability negotiation for the DSA indexer top-k.

This module is deliberately free of torch, CUDA and sglang runtime imports so
that the dtype/alignment negotiation and the BF16x2 packed histogram extraction
can be unit-tested in a minimal pytest-only environment.

It mirrors the predicates used by the JIT kernels:

* ``kpool_topk_transform.cuh`` (``takes_bf16_logits``)
* ``dsa_gfx950/topk_transform.cuh`` (``takes_bf16_logits``)
* ``topk_histogram_4096.cuh`` (``extract_coarse_bins_bf16x2``)
"""

from __future__ import annotations

from typing import Tuple

# DLDataType codes, kept as plain ints so dlpack need not be importable.
KDL_FLOAT = 2
KDL_BFLOAT = 4

# Elements per 16-byte vector, derived from the element width rather than
# hardcoded to the FP32 assumption.
def vec_width(elem_bytes: int) -> int:
    if elem_bytes <= 0 or 16 % elem_bytes != 0:
        raise ValueError(f"unsupported element width: {elem_bytes}")
    return 16 // elem_bytes


def takes_bf16_logits_for(
    dtype_code: int,
    dtype_bits: int,
    dtype_lanes: int,
    row_stride: int,
) -> bool:
    """Host-side mirror of the kernels' BF16 native-read predicate.

    BF16 logits are read natively only when the row stride keeps every row
    16-byte aligned; otherwise the caller upcasts and takes the FP32 kernel.
    """
    return (
        dtype_code == KDL_BFLOAT
        and dtype_bits == 16
        and dtype_lanes == 1
        and row_stride % vec_width(2) == 0
    )


def takes_fp32_logits_for(dtype_code: int, dtype_bits: int, dtype_lanes: int) -> bool:
    """FP32 logits are always readable; named so the dispatcher can negotiate."""
    return dtype_code == KDL_FLOAT and dtype_bits == 32 and dtype_lanes == 1


def resolve_logits_dtype(
    dtype_code: int,
    dtype_bits: int,
    dtype_lanes: int,
    row_stride: int,
) -> str:
    """Return ``"bf16"`` or ``"fp32"`` for the top-k kernel to dispatch on.

    Anything that is not a natively readable BF16 tensor falls back to FP32,
    which is the pre-existing behaviour.
    """
    if takes_bf16_logits_for(dtype_code, dtype_bits, dtype_lanes, row_stride):
        return "bf16"
    if takes_fp32_logits_for(dtype_code, dtype_bits, dtype_lanes):
        return "fp32"
    # Unknown dtype: the caller is expected to upcast before calling us.
    return "fp32"


def _order_key16(bits: int) -> int:
    """Ordered 16-bit key of a BF16 bit pattern (monotonic in the value)."""
    bits &= 0xFFFF
    if bits & 0x8000:
        return (~bits) & 0xFFFF
    return (bits | 0x8000) & 0xFFFF


def host_convert_bf16_to_uint8(bits: int) -> int:
    """Coarse 8-bit bin of a BF16 bit pattern, matching the kernel."""
    return (_order_key16(bits) >> 8) & 0xFF


def host_convert_bf16x2_to_uint8x2(word: int) -> Tuple[int, int]:
    """Coarse bins of the two BF16 scores packed in ``word`` (low half first).

    This is the host-side mirror of ``convert_bf16x2_to_uint8x2`` in
    ``kpool_topk_transform.cuh``; it must agree with the per-element form.
    """
    word &= 0xFFFFFFFF
    lo = word & 0xFFFF
    hi = (word >> 16) & 0xFFFF
    return host_convert_bf16_to_uint8(lo), host_convert_bf16_to_uint8(hi)


def pack_bf16x2(lo_bits: int, hi_bits: int) -> int:
    """Pack two BF16 bit patterns into one 32-bit word (low half first)."""
    return ((hi_bits & 0xFFFF) << 16) | (lo_bits & 0xFFFF)
