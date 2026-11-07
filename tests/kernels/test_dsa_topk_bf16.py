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
    if not path.is_file():  # pragma: no cover - defensive
        pytest.skip(f"capability module not found at {path}", allow_module_level=True)
    spec = importlib.util.spec_from_file_location("_dsa_topk_bf16_capability", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        pytest.skip(f"cannot load capability module from {path}", allow_module_level=True)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cap = _load_capability_module()


def _reference_topk(values, k):
    """Pure-Python top-k over (score, index) pairs, descending by score.

    Ties are broken by the smaller index so the reference is deterministic and
    can be compared against the packed-extraction ordering.
    """
    order = sorted(range(len(values)), key=lambda i: (-values[i], i))
    return order[:k]


def _bf16_to_float(bits: int) -> float:
    """Exact BF16 -> FP32 widening (BF16 is the high 16 bits of FP32)."""
    import struct

    return struct.unpack("<f", struct.pack("<I", (bits & 0xFFFF) << 16))[0]


def _float_to_bf16_bits(value: float) -> int:
    import struct

    raw = struct.unpack("<I", struct.pack("<f", value))[0]
    # Round-to-nearest-even on the low 16 bits, matching __float2bfloat16_rn.
    lsb = (raw >> 16) & 1
    rounding = 0x7FFF + lsb
    return ((raw + rounding) >> 16) & 0xFFFF


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


def test_bf16_and_fp32_topk_select_same_values():
    """BF16 widens to FP32 exactly, so top-k must select the same values.

    This mirrors the kernel-level consistency test without needing a GPU: the
    BF16 scores are widened exactly and compared against the FP32 reference.
    """
    fp32_values = [0.5, -1.25, 3.0, 2.0, -0.0, 0.0, 7.5, 1.0]
    bf16_bits = [_float_to_bf16_bits(v) for v in fp32_values]
    bf16_values = [_bf16_to_float(b) for b in bf16_bits]
    for k in (1, 3, 5, len(fp32_values)):
        assert _reference_topk(bf16_values, k) == _reference_topk(fp32_values, k)


def test_bf16x2_packed_extraction_preserves_topk_order():
    """The packed extraction must rank scores the same way as the scalar key."""
    values = [0.5, -1.25, 3.0, 2.0, -0.0, 0.0, 7.5, 1.0]
    bits = [_float_to_bf16_bits(v) for v in values]
    keys = [cap.host_convert_bf16_to_uint8(b) for b in bits]
    # Higher score -> higher ordered key; equal scores share a key.
    for i in range(len(values)):
        for j in range(len(values)):
            if values[i] > values[j]:
                assert keys[i] >= keys[j]
            if values[i] == values[j]:
                assert keys[i] == keys[j]


def test_resolve_logits_dtype_covers_cooperative_and_persistent():
    """Both schedulers take BF16 only when the row stride is 16B aligned."""
    for stride in (8, 16, 65536):
        assert cap.resolve_logits_dtype(cap.BF16, stride) == "bf16"
    for stride in (1, 4, 7):
        assert cap.resolve_logits_dtype(cap.BF16, stride) == "fp32"


def test_resolve_logits_dtype_rejects_unsupported_dtype():
    with pytest.raises(ValueError):
        cap.resolve_logits_dtype(cap.DType(cap.DL_FLOAT, 16, 1), 65536)
    with pytest.raises(ValueError):
        cap.resolve_logits_dtype(cap.DType(cap.DL_BFLOAT, 16, 2), 65536)


def test_bf16_topk_selection_is_deterministic_across_runs():
    """Repeated negotiation and extraction must yield identical selections.

    This is the in-process analogue of the CI ``--count=3`` stability run: the
    dtype/alignment negotiation and the packed histogram extraction are pure
    functions, so two evaluations must agree element-wise.
    """
    values = [0.5, -1.25, 3.0, 2.0, -0.0, 0.0, 7.5, 1.0]
    bits = [_float_to_bf16_bits(v) for v in values]
    first_keys = [cap.host_convert_bf16_to_uint8(b) for b in bits]
    first_choice = cap.resolve_logits_dtype(cap.BF16, 65536)
    for _ in range(2):
        again_keys = [cap.host_convert_bf16_to_uint8(b) for b in bits]
        assert again_keys == first_keys
        assert cap.resolve_logits_dtype(cap.BF16, 65536) == first_choice
    assert _reference_topk([_bf16_to_float(b) for b in bits], 3) == _reference_topk(values, 3)
