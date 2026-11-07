"""Host-side BF16 capability negotiation for the DSA indexer top-k kernels.

This module mirrors, in pure Python, the predicates the CUDA/HIP kernels use to
decide whether they can read BF16 logits natively instead of upcasting them to
FP32 first.  Keeping the logic here (free of torch/CUDA imports) lets the
negotiation and its fallback be unit-tested in a minimal pytest+numpy job.

The rules are deliberately identical to the device-side checks:

* ``cooperative_topk`` / ``persistent_topk`` read BF16 when the row stride keeps
  every row 16-byte aligned, i.e. ``stride % (16 // sizeof(InT)) == 0``.
* ``kpool_topk_transform`` reads BF16 when the row stride is a multiple of the
  16-byte vector width (8 BF16 elements).
* ``dsa_gfx950`` top-k reads BF16 when the row stride is a multiple of 8.

Anything else falls back to the FP32 kernel, which the caller reaches by
upcasting the logits once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# DLDataType codes we care about; kept as plain ints so this module never has to
# import dlpack/torch.
DL_FLOAT = 2
DL_BFLOAT = 16

# Element widths in bytes.
_FP32_BYTES = 4
_BF16_BYTES = 2

# 16-byte TMA/vector unit shared by every kernel below.
_TMA_BYTES = 16


def tma_align(element_bytes: int) -> int:
    """Elements per 16-byte TMA unit for an element of ``element_bytes``."""
    if element_bytes <= 0 or _TMA_BYTES % element_bytes != 0:
        raise ValueError(f"unsupported element width: {element_bytes}")
    return _TMA_BYTES // element_bytes


def is_bf16(dtype_code: int, dtype_bits: int, dtype_lanes: int) -> bool:
    return dtype_code == DL_BFLOAT and dtype_bits == 16 and dtype_lanes == 1


def is_fp32(dtype_code: int, dtype_bits: int, dtype_lanes: int) -> bool:
    return dtype_code == DL_FLOAT and dtype_bits == 32 and dtype_lanes == 1


def bf16_row_aligned(row_stride: int, element_bytes: int = _BF16_BYTES) -> bool:
    """True when every row starts on a 16-byte boundary for this element width."""
    if row_stride < 0:
        return False
    return row_stride % tma_align(element_bytes) == 0


@dataclass(frozen=True)
class TopKBackend:
    """A top-k backend and the dtypes it can read natively."""

    name: str
    takes_bf16: bool
    takes_fp32: bool = True


# Backends the dispatcher may pick.  ``cooperative`` and ``persistent`` share the
# same 16-byte alignment rule; ``deep_select`` and the gfx950 scatter kernel
# read BF16 natively as well.
BACKENDS = {
    "cooperative": TopKBackend("cooperative", takes_bf16=True),
    "persistent": TopKBackend("persistent", takes_bf16=True),
    "deep_select": TopKBackend("deep_select", takes_bf16=True),
    "gfx950_scatter": TopKBackend("gfx950_scatter", takes_bf16=True),
    # A backend that only ever learned FP32; used to exercise the fallback.
    "fp32_only": TopKBackend("fp32_only", takes_bf16=False),
}


@dataclass(frozen=True)
class Resolution:
    """Outcome of dtype negotiation for one launch."""

    backend: str
    dtype: str  # "bf16" or "fp32"
    upcast: bool  # True when the caller must upcast before the launch
    reason: str


def resolve_topk_dtype(
    backend: str,
    dtype_code: int,
    dtype_bits: int,
    dtype_lanes: int,
    row_stride: int,
    *,
    num_rows: int = 1,
    max_cooperative_rows: int = 64,
) -> Resolution:
    """Decide whether ``backend`` reads the logits natively or needs FP32.

    ``auto`` is resolved to ``cooperative`` for up to ``max_cooperative_rows``
    rows and to ``persistent`` above that, matching the scheduler.  A backend
    that cannot read BF16, or a BF16 row that is not 16-byte aligned, falls back
    to FP32 with ``upcast=True``.
    """
    if backend == "auto":
        backend = "cooperative" if num_rows <= max_cooperative_rows else "persistent"
    if backend not in BACKENDS:
        raise ValueError(f"unknown top-k backend: {backend}")
    spec = BACKENDS[backend]

    if is_fp32(dtype_code, dtype_bits, dtype_lanes):
        return Resolution(backend, "fp32", False, "fp32 logits are always readable")

    if not is_bf16(dtype_code, dtype_bits, dtype_lanes):
        raise ValueError(
            "logits must be float32 or bfloat16, got "
            f"code={dtype_code} bits={dtype_bits} lanes={dtype_lanes}"
        )

    if not spec.takes_bf16:
        return Resolution(backend, "fp32", True, f"{backend} does not take bf16 logits")

    if not bf16_row_aligned(row_stride):
        return Resolution(
            backend,
            "fp32",
            True,
            f"bf16 row stride {row_stride} is not a multiple of {tma_align(_BF16_BYTES)}",
        )

    return Resolution(backend, "bf16", False, f"{backend} reads bf16 logits natively")


def takes_bf16_logits(
    dtype_code: int,
    dtype_bits: int,
    dtype_lanes: int,
    row_stride: int,
    *,
    backend: str = "cooperative",
    num_rows: int = 1,
) -> bool:
    """Convenience predicate mirroring the device-side ``takes_bf16_logits``."""
    return resolve_topk_dtype(
        backend, dtype_code, dtype_bits, dtype_lanes, row_stride, num_rows=num_rows
    ).dtype == "bf16"


def paged_mqa_logits_ok(
    dtype_code: int,
    dtype_bits: int,
    dtype_lanes: int,
    row_stride: int,
    seq_len: int,
    page_size: int,
) -> bool:
    """Boundary check for the paged MQA logits kernel.

    The kernel walks whole pages, so ``seq_len`` may be shorter than the row but
    the row must still be 16-byte aligned for the element width.  An empty row
    (``seq_len == 0``) is valid and selects nothing.
    """
    if seq_len < 0 or page_size <= 0:
        return False
    if is_bf16(dtype_code, dtype_bits, dtype_lanes):
        return bf16_row_aligned(row_stride)
    if is_fp32(dtype_code, dtype_bits, dtype_lanes):
        return row_stride % tma_align(_FP32_BYTES) == 0
    return False


def bf16x2_to_uint8x2(word: int) -> tuple[int, int]:
    """Host mirror of ``convert_bf16x2_to_uint8x2``.

    ``word`` packs two BF16 bit patterns, low half first.  Returns the two
    ordered 8-bit keys the histogram uses.
    """
    if not 0 <= word <= 0xFFFFFFFF:
        raise ValueError(f"word out of range: {word}")
    lo = word & 0xFFFF
    hi = (word >> 16) & 0xFFFF
    return _bf16_bits_to_uint8(lo), _bf16_bits_to_uint8(hi)


def _bf16_bits_to_uint8(bits: int) -> int:
    key = (~bits) & 0xFFFF if (bits & 0x8000) else (bits | 0x8000)
    return (key >> 8) & 0xFF


def bf16_to_uint8(bits: int) -> int:
    """Host mirror of ``convert_bf16_to_uint8`` for a single BF16 bit pattern."""
    if not 0 <= bits <= 0xFFFF:
        raise ValueError(f"bits out of range: {bits}")
    return _bf16_bits_to_uint8(bits)


def fp32_to_uint8(value: float) -> int:
    """Host mirror of ``convert_to_uint8``: round to fp16, then order-key it."""
    bits = _float_to_half_bits(value)
    key = (~bits) & 0xFFFF if (bits & 0x8000) else (bits | 0x8000)
    return (key >> 8) & 0xFF


def _float_to_half_bits(value: float) -> int:
    """Round a Python float to IEEE fp16 and return its 16-bit pattern.

    Implemented with ``struct`` so this module stays free of numpy/torch.
    """
    import struct

    packed = struct.pack("<e", value)
    return struct.unpack("<H", packed)[0]


def select_topk_indices(scores: list[float], k: int) -> list[int]:
    """Reference top-k used by the tests: descending value, ties by index."""
    if k < 0:
        raise ValueError("k must be non-negative")
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    return order[:k]


def optional_int(value: Optional[int]) -> Optional[int]:
    """Small helper kept for symmetry with the kernel's optional arguments."""
    return value
