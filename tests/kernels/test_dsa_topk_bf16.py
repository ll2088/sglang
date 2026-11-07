"""Standalone tests for the BF16 indexer top-k capability negotiation.

These tests only use the standard library and the pure-Python helper module,
so they can be collected and run without torch, CUDA or the full sglang
runtime dependency set.  They cover the dtype/alignment negotiation, the
FP32 fallback branch and the BF16x2 packed histogram extraction.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_capability_module():
    path = _REPO_ROOT / "python/sglang/kernels/jit/csrc/dsa/topk_bf16_capability.py"
    spec = importlib.util.spec_from_file_location("_dsa_topk_bf16_capability", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load capability module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cap = _load_capability_module()


def test_vec_width_is_derived_from_element_width():
    assert cap.vec_width(4) == 4
    assert cap.vec_width(2) == 8
    with pytest.raises(ValueError):
        cap.vec_width(3)


def test_takes_bf16_logits_requires_16_byte_alignment():
    assert cap.takes_bf16_logits(cap.BF16, 65536)
    assert cap.takes_bf16_logits(cap.BF16, 8)
    assert not cap.takes_bf16_logits(cap.BF16, 4)
    assert not cap.takes_bf16_logits(cap.BF16, 7)


def test_takes_bf16_logits_rejects_other_dtypes():
    assert not cap.takes_bf16_logits(cap.FP32, 65536)
    assert not cap.takes_bf16_logits(cap.DType(cap.DL_BFLOAT, 16, 2), 65536)
    assert not cap.takes_bf16_logits(cap.DType(cap.DL_BFLOAT, 32, 1), 65536)


def test_takes_fp32_logits_always_true_for_fp32():
    assert cap.takes_fp32_logits(cap.FP32)
    assert not cap.takes_fp32_logits(cap.BF16)


def test_resolve_logits_dtype_negotiates_and_falls_back():
    assert cap.resolve_logits_dtype(cap.BF16, 65536) == "bf16"
    # Misaligned BF16 rows fall back to the FP32 kernel after upcast.
    assert cap.resolve_logits_dtype(cap.BF16, 4) == "fp32"
    assert cap.resolve_logits_dtype(cap.FP32, 65536) == "fp32"
    with pytest.raises(ValueError):
        cap.resolve_logits_dtype(cap.DType(cap.DL_FLOAT, 16, 1), 65536)


def test_bf16x2_packed_extraction_matches_scalar():
    # +1.0, -1.0, +0.0, -0.0 as BF16 bit patterns.
    cases = [
        (0x3F80, 0xBF80),
        (0x0000, 0x8000),
        (0x7F7F, 0xFF7F),
        (0x3F80, 0x3F80),
    ]
    for lo, hi in cases:
        word = cap.pack_bf16x2(lo, hi)
        got = cap.host_convert_bf16x2_to_uint8x2(word)
        want = (cap.host_convert_bf16_to_uint8(lo), cap.host_convert_bf16_to_uint8(hi))
        assert got == want


def test_bf16x2_packed_extraction_is_order_preserving():
    # Ordered keys must be monotonic in the score for finite BF16 values.
    values = [0x0000, 0x3F80, 0x4000, 0x7F7F, 0x8000, 0xBF80, 0xC000, 0xFF7F]
    keys = [cap.host_convert_bf16_to_uint8(v) for v in values]
    assert keys == sorted(keys)


def test_bf16x2_packed_extraction_handles_high_half():
    word = cap.pack_bf16x2(0x3F80, 0xBF80)
    lo, hi = cap.host_convert_bf16x2_to_uint8x2(word)
    assert lo == cap.host_convert_bf16_to_uint8(0x3F80)
    assert hi == cap.host_convert_bf16_to_uint8(0xBF80)
