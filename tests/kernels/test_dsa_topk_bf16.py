"""Standalone tests for the DSA indexer top-k BF16 capability negotiation.

These tests are dependency-light on purpose: they only need the standard
library and the pure-Python helper module, so they can run in CI without
installing the sglang meta-package or any CUDA/torch dependency.
"""

from __future__ import annotations

import struct

import pytest

from dsa_topk_bf16_capability import (
    KDL_BFLOAT,
    KDL_FLOAT,
    host_convert_bf16_to_uint8,
    host_convert_bf16x2_to_uint8x2,
    pack_bf16x2,
    resolve_logits_dtype,
    takes_bf16_logits_for,
    takes_fp32_logits_for,
    vec_width,
)


def _bf16_bits(value: float) -> int:
    """Round a Python float to BF16 and return its 16-bit pattern."""
    packed = struct.pack("<f", value)
    (word,) = struct.unpack("<I", packed)
    # Round-to-nearest-even on the low 16 bits.
    lsb = (word >> 16) & 1
    word = (word + 0x7FFF + lsb) & 0xFFFFFFFF
    return (word >> 16) & 0xFFFF


# ---------------------------------------------------------------------------
# Element-width parameterisation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "elem_bytes,expected",
    [(4, 4), (2, 8), (1, 16)],
)
def test_vec_width_is_derived_from_element_width(elem_bytes: int, expected: int) -> None:
    assert vec_width(elem_bytes) == expected


def test_vec_width_rejects_unsupported_width() -> None:
    with pytest.raises(ValueError):
        vec_width(3)


# ---------------------------------------------------------------------------
# Backend dtype capability negotiation and fallback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("row_stride", [8, 16, 1024, 65536])
def test_bf16_native_read_when_row_is_16b_aligned(row_stride: int) -> None:
    assert takes_bf16_logits_for(KDL_BFLOAT, 16, 1, row_stride)
    assert resolve_logits_dtype(KDL_BFLOAT, 16, 1, row_stride) == "bf16"


@pytest.mark.parametrize("row_stride", [1, 2, 3, 7, 9, 15])
def test_bf16_falls_back_when_row_is_not_16b_aligned(row_stride: int) -> None:
    assert not takes_bf16_logits_for(KDL_BFLOAT, 16, 1, row_stride)
    # Misaligned BF16 must fall back to the FP32 path, not be read natively.
    assert resolve_logits_dtype(KDL_BFLOAT, 16, 1, row_stride) == "fp32"


@pytest.mark.parametrize(
    "code,bits,lanes",
    [
        (KDL_FLOAT, 32, 1),
        (KDL_BFLOAT, 32, 1),
        (KDL_BFLOAT, 16, 2),
        (KDL_FLOAT, 16, 1),
    ],
)
def test_non_bf16_dtypes_do_not_take_bf16_path(code: int, bits: int, lanes: int) -> None:
    assert not takes_bf16_logits_for(code, bits, lanes, 1024)
    assert resolve_logits_dtype(code, bits, lanes, 1024) == "fp32"


def test_fp32_is_always_readable() -> None:
    assert takes_fp32_logits_for(KDL_FLOAT, 32, 1)
    assert not takes_fp32_logits_for(KDL_BFLOAT, 16, 1)


# ---------------------------------------------------------------------------
# BF16x2 packed histogram extraction
# ---------------------------------------------------------------------------


def test_bf16x2_matches_per_element_extraction() -> None:
    values = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        -0.5,
        3.14159,
        -2.71828,
        1e-8,
        -1e-8,
        1e8,
        -1e8,
    ]
    for lo in values:
        for hi in values:
            lo_bits = _bf16_bits(lo)
            hi_bits = _bf16_bits(hi)
            word = pack_bf16x2(lo_bits, hi_bits)
            got = host_convert_bf16x2_to_uint8x2(word)
            assert got == (
                host_convert_bf16_to_uint8(lo_bits),
                host_convert_bf16_to_uint8(hi_bits),
            )


def test_bf16x2_preserves_ordering() -> None:
    # The ordered key must be monotonic in the value, so the coarse bin of a
    # larger score is never smaller than that of a smaller score.
    ordered = sorted(
        [
            -1e8,
            -1.0,
            -0.5,
            -0.0,
            0.0,
            0.5,
            1.0,
            1e8,
        ]
    )
    bins = [host_convert_bf16_to_uint8(_bf16_bits(v)) for v in ordered]
    assert bins == sorted(bins)


def test_bf16x2_pack_roundtrip() -> None:
    lo_bits = _bf16_bits(1.5)
    hi_bits = _bf16_bits(-2.25)
    word = pack_bf16x2(lo_bits, hi_bits)
    assert (word & 0xFFFF) == lo_bits
    assert ((word >> 16) & 0xFFFF) == hi_bits
