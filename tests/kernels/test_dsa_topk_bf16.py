"""Standalone tests for the DSA top-k BF16 capability negotiation.

These tests only depend on numpy and pytest so they can run in a minimal CI
job without installing the full sglang dependency set or a CUDA toolchain.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from sglang.kernels.dsa_topk_bf16_capability import (
    DTypeInfo,
    bf16_bits_to_coarse_bin,
    bf16_bits_to_key,
    bf16x2_to_coarse_bins,
    resolve_backend,
    select_logits_dtype,
    takes_bf16_logits,
    takes_fp32_logits,
)

BF16 = DTypeInfo(code="bfloat", bits=16)
FP32 = DTypeInfo(code="float", bits=32)


def _float_to_bf16_bits(x: float) -> int:
    """Round-to-nearest-even float32 -> bfloat16, returned as raw bits."""
    f32 = struct.unpack("<I", struct.pack("<f", float(x)))[0]
    lsb = (f32 >> 16) & 1
    rounding_bias = 0x7FFF + lsb
    return ((f32 + rounding_bias) >> 16) & 0xFFFF


def _reference_coarse_bin(x: float) -> int:
    bits = _float_to_bf16_bits(x)
    key = (~bits) & 0xFFFF if bits & 0x8000 else (bits | 0x8000) & 0xFFFF
    return key >> 8


@pytest.mark.parametrize(
    "values",
    [
        [0.0, -0.0, 1.0, -1.0],
        [0.5, -0.5, 2.0, -2.0],
        [1e-3, -1e-3, 1e3, -1e3],
        [3.14159, -3.14159, 0.125, -0.125],
    ],
)
def test_bf16x2_packed_bins_match_scalar(values):
    for i in range(0, len(values), 2):
        lo = _float_to_bf16_bits(values[i])
        hi = _float_to_bf16_bits(values[i + 1])
        word = lo | (hi << 16)
        got = bf16x2_to_coarse_bins(word)
        want = (_reference_coarse_bin(values[i]), _reference_coarse_bin(values[i + 1]))
        assert got == want


def test_bf16_key_is_order_preserving():
    xs = np.linspace(-8.0, 8.0, 257)
    keys = [bf16_bits_to_key(_float_to_bf16_bits(float(x))) for x in xs]
    assert keys == sorted(keys)


def test_bf16_coarse_bin_matches_key_shift():
    for x in (-2.0, -0.5, 0.0, 0.25, 1.5, 4.0):
        bits = _float_to_bf16_bits(x)
        assert bf16_bits_to_coarse_bin(bits) == bf16_bits_to_key(bits) >> 8


def test_takes_bf16_requires_alignment():
    assert takes_bf16_logits(BF16, 65536)
    assert takes_bf16_logits(BF16, 8)
    assert not takes_bf16_logits(BF16, 7)
    assert not takes_bf16_logits(BF16, 4)
    assert not takes_bf16_logits(FP32, 65536)


def test_takes_fp32():
    assert takes_fp32_logits(FP32)
    assert not takes_fp32_logits(BF16)


@pytest.mark.parametrize("num_rows", [1, 4, 8, 16, 33, 64])
def test_auto_picks_cooperative_for_small_batches(num_rows):
    backend, use_bf16 = resolve_backend("auto", BF16, 65536, num_rows)
    assert backend == "cooperative"
    assert use_bf16 is True


@pytest.mark.parametrize("num_rows", [65, 128, 1024])
def test_auto_picks_persistent_for_large_batches(num_rows):
    backend, use_bf16 = resolve_backend("auto", BF16, 65536, num_rows)
    assert backend == "persistent"
    assert use_bf16 is True


def test_unaligned_bf16_falls_back_to_fp32():
    backend, dtype = select_logits_dtype("auto", BF16, 7, 16)
    assert backend == "cooperative"
    assert dtype == FP32


def test_backend_without_bf16_falls_back_to_fp32():
    backend, dtype = select_logits_dtype("deep_select", BF16, 65536, 16)
    assert backend == "deep_select"
    assert dtype == FP32


def test_fp32_input_stays_fp32():
    backend, dtype = select_logits_dtype("auto", FP32, 65536, 16)
    assert backend == "cooperative"
    assert dtype == FP32


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        resolve_backend("does_not_exist", BF16, 65536, 16)


def test_unsupported_dtype_raises():
    with pytest.raises(ValueError):
        resolve_backend("auto", DTypeInfo(code="float", bits=16), 65536, 16)


def test_repeated_resolution_is_deterministic():
    first = select_logits_dtype("auto", BF16, 65536, 16)
    for _ in range(3):
        assert select_logits_dtype("auto", BF16, 65536, 16) == first
