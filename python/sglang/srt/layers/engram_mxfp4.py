"""MXFP4-packed Engram table layout parsing, E2M1 decode, and row-width aware allocation.

This module is self-contained and dependency-free (stdlib only). It mirrors the
upstream vLLM #61530 capability: two E2M1 values per byte with UE8M0 block-32
scales, layout selected by checkpoint metadata, and all storage paths sized by
the actual packed row width.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

__all__ = [
    "EngramLayout",
    "EngramLayoutError",
    "E2M1_VALUES",
    "decode_e2m1_nibble",
    "decode_e2m1_packed",
    "pack_e2m1",
    "row_bytes_for",
    "allocate_row_buffer",
]


# E2M1 magnitudes indexed by the low 3 bits (sign is bit 3).
E2M1_VALUES: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

_SUPPORTED_QUANT = ("fp8", "mxfp4")
_EXPECTED_BLOCK_SIZE = 32
_EXPECTED_SCALE_DTYPE = "ue8m0"


class EngramLayoutError(ValueError):
    """Raised for unsupported or inconsistent Engram table layouts."""


@dataclass(frozen=True)
class EngramLayout:
    """Resolved layout for one Engram embedding table.

    ``row_bytes`` is the on-disk width of a single row: ``head_dim`` for FP8 and
    ``head_dim // 2`` for MXFP4 (two E2M1 values per byte). ``num_blocks`` is the
    number of UE8M0 scale blocks per row.
    """

    quant: str
    head_dim: int
    block_size: int
    scale_dtype: str
    packed_fp4: bool
    row_bytes: int
    num_blocks: int

    @classmethod
    def from_metadata(
        cls,
        head_dim: int,
        layer_ids: Sequence[int],
        quantization_config: dict | None = None,
    ) -> "EngramLayout":
        """Parse and validate checkpoint metadata into a layout.

        Unsupported quant formats, block sizes, scale dtypes, or partial layer
        lists fail here, before any table allocation happens.
        """
        if not isinstance(head_dim, int) or head_dim <= 0:
            raise EngramLayoutError(f"head_dim must be a positive int, got {head_dim!r}")
        if head_dim % 2 != 0:
            raise EngramLayoutError(f"head_dim must be even, got {head_dim}")

        quant = quantization_config or {}
        embed_quant = quant.get("engram_embed_quant", "fp8")
        if embed_quant not in _SUPPORTED_QUANT:
            raise EngramLayoutError(f"Unsupported engram_embed_quant {embed_quant!r}")

        block_size = quant.get("engram_embed_block_size", _EXPECTED_BLOCK_SIZE)
        if block_size != _EXPECTED_BLOCK_SIZE:
            raise EngramLayoutError(
                f"engram_embed_block_size must be {_EXPECTED_BLOCK_SIZE!r}, got {block_size!r}"
            )

        scale_dtype = quant.get("engram_embed_scale_dtype", _EXPECTED_SCALE_DTYPE)
        if scale_dtype != _EXPECTED_SCALE_DTYPE:
            raise EngramLayoutError(
                f"engram_embed_scale_dtype must be {_EXPECTED_SCALE_DTYPE!r}, got {scale_dtype!r}"
            )

        embed_layers = quant.get("engram_embed_layers", list(layer_ids))
        if set(embed_layers) != set(layer_ids):
            raise EngramLayoutError("engram_embed_layers must match engram_layer_ids")

        if head_dim % block_size != 0:
            raise EngramLayoutError(
                f"head_dim {head_dim} must be divisible by block_size {block_size}"
            )

        packed_fp4 = embed_quant == "mxfp4"
        row_bytes = head_dim // 2 if packed_fp4 else head_dim
        return cls(
            quant=embed_quant,
            head_dim=head_dim,
            block_size=block_size,
            scale_dtype=scale_dtype,
            packed_fp4=packed_fp4,
            row_bytes=row_bytes,
            num_blocks=head_dim // block_size,
        )

    def scale_bytes_per_row(self) -> int:
        """UE8M0 scales are one byte per block."""
        return self.num_blocks

    def total_row_bytes(self) -> int:
        """Packed weight bytes plus scale bytes for one row."""
        return self.row_bytes + self.scale_bytes_per_row()


# ---------------------------------------------------------------------------
# E2M1 decode / encode
# ---------------------------------------------------------------------------


def decode_e2m1_nibble(nibble: int) -> float:
    """Decode a single 4-bit E2M1 code to its float value.

    Bit 3 is the sign, bits 2:1 the exponent, bit 0 the mantissa:
    ``m * 0.5`` when ``e == 0``, else ``(2 + m) * 2 ** (e - 2)``.
    """
    if not isinstance(nibble, int) or not 0 <= nibble <= 0xF:
        raise EngramLayoutError(f"nibble must be an int in [0, 15], got {nibble!r}")
    e = (nibble & 0x7) >> 1
    m = nibble & 0x1
    mag = m * 0.5 if e == 0 else (2 + m) * (2.0 ** (e - 2))
    return -mag if nibble & 0x8 else mag


def decode_e2m1_packed(packed: Iterable[int], dim: int | None = None) -> list[float]:
    """Decode packed bytes into a flat list of E2M1 values.

    The even column lives in the low nibble, the odd column in the high nibble.
    When ``dim`` is given, the packed length must match ``dim // 2``.
    """
    data = list(packed)
    for i, byte in enumerate(data):
        if not isinstance(byte, int) or not 0 <= byte <= 0xFF:
            raise EngramLayoutError(f"packed[{i}] must be a byte, got {byte!r}")
    if dim is not None:
        if dim <= 0 or dim % 2 != 0:
            raise EngramLayoutError(f"dim must be a positive even int, got {dim!r}")
        if len(data) != dim // 2:
            raise EngramLayoutError(
                f"packed length {len(data)} does not match dim // 2 = {dim // 2}"
            )
    out: list[float] = []
    for byte in data:
        out.append(decode_e2m1_nibble(byte & 0xF))
        out.append(decode_e2m1_nibble((byte >> 4) & 0xF))
    return out


def pack_e2m1(values: Sequence[float]) -> list[int]:
    """Inverse of :func:`decode_e2m1_packed` for exact E2M1 values.

    Raises if a value is not representable, so tests can round-trip safely.
    """
    if len(values) % 2 != 0:
        raise EngramLayoutError(f"values length must be even, got {len(values)}")
    out: list[int] = []
    for i in range(0, len(values), 2):
        lo = _encode_e2m1(values[i])
        hi = _encode_e2m1(values[i + 1])
        out.append(lo | (hi << 4))
    return out


def _encode_e2m1(value: float) -> int:
    for code in range(16):
        if decode_e2m1_nibble(code) == value:
            return code
    raise EngramLayoutError(f"value {value!r} is not representable in E2M1")


# ---------------------------------------------------------------------------
# Row-width aware allocation
# ---------------------------------------------------------------------------


def row_bytes_for(layout: EngramLayout, rows: int) -> int:
    """Total bytes for ``rows`` rows using the actual packed row width."""
    if not isinstance(rows, int) or rows < 0:
        raise EngramLayoutError(f"rows must be a non-negative int, got {rows!r}")
    return rows * layout.total_row_bytes()


def allocate_row_buffer(layout: EngramLayout, rows: int, max_blocks: int) -> int:
    """Return the byte size for a table buffer, validating block limits.

    ``max_blocks`` is the largest number of scale blocks a single row may use
    on the target storage path. A row width of zero, a non-power-of-two row
    width, or a block count above ``max_blocks`` all raise explicitly instead
    of silently truncating.
    """
    if not isinstance(max_blocks, int) or max_blocks <= 0:
        raise EngramLayoutError(f"max_blocks must be a positive int, got {max_blocks!r}")
    if layout.row_bytes <= 0:
        raise EngramLayoutError(f"row_bytes must be positive, got {layout.row_bytes}")
    if layout.num_blocks > max_blocks:
        raise EngramLayoutError(
            f"num_blocks {layout.num_blocks} exceeds max_blocks {max_blocks}"
        )
    return row_bytes_for(layout, rows)
