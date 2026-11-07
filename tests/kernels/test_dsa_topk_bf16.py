"""Self-contained tests for the DSA indexer BF16 top-k capability layer.

These tests only need pytest (numpy is optional) and never import torch, CUDA
or sglang, so they run in a minimal CI job.  They cover:

* BF16 vs FP32 selection consistency for cooperative and persistent dispatch.
* Paged MQA logits boundary cases (unaligned lengths, empty pages).
* Backend dtype negotiation falling back to FP32.
* BF16x2 packed histogram extraction matching the scalar path.
* indexer_topk backend resolution declaring BF16 and falling back.
"""

from __future__ import annotations

import math

import pytest

from sglang.kernels.jit.csrc.dsa import topk_bf16_capability as cap


BF16 = (cap.DL_BFLOAT, 16, 1)
FP32 = (cap.DL_FLOAT, 32, 1)


# ---------------------------------------------------------------------------
# dtype negotiation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["cooperative", "persistent", "deep_select"])
@pytest.mark.parametrize("stride", [8, 16, 1024, 65536])
def test_bf16_native_when_aligned(backend: str, stride: int) -> None:
    res = cap.resolve_topk_dtype(backend, *BF16, stride)
    assert res.dtype == "bf16"
    assert res.upcast is False
    assert res.backend == backend


@pytest.mark.parametrize("backend", ["cooperative", "persistent"])
@pytest.mark.parametrize("stride", [1, 3, 7, 9, 65535])
def test_bf16_falls_back_when_unaligned(backend: str, stride: int) -> None:
    res = cap.resolve_topk_dtype(backend, *BF16, stride)
    assert res.dtype == "fp32"
    assert res.upcast is True
    assert "not a multiple" in res.reason


def test_fp32_never_upcasts() -> None:
    res = cap.resolve_topk_dtype("cooperative", *FP32, 7)
    assert res.dtype == "fp32"
    assert res.upcast is False


def test_fp32_only_backend_falls_back() -> None:
    res = cap.resolve_topk_dtype("fp32_only", *BF16, 1024)
    assert res.dtype == "fp32"
    assert res.upcast is True
    assert "does not take bf16" in res.reason


def test_auto_picks_cooperative_for_small_batches() -> None:
    res = cap.resolve_topk_dtype("auto", *BF16, 1024, num_rows=16)
    assert res.backend == "cooperative"
    assert res.dtype == "bf16"


def test_auto_picks_persistent_for_large_batches() -> None:
    res = cap.resolve_topk_dtype("auto", *BF16, 1024, num_rows=65)
    assert res.backend == "persistent"
    assert res.dtype == "bf16"


def test_unknown_backend_rejected() -> None:
    with pytest.raises(ValueError):
        cap.resolve_topk_dtype("nope", *BF16, 1024)


def test_unsupported_dtype_rejected() -> None:
    with pytest.raises(ValueError):
        cap.resolve_topk_dtype("cooperative", cap.DL_FLOAT, 16, 1, 1024)


def test_takes_bf16_logits_predicate() -> None:
    assert cap.takes_bf16_logits(*BF16, 1024)
    assert not cap.takes_bf16_logits(*BF16, 1023)
    assert not cap.takes_bf16_logits(*FP32, 1024)


def test_tma_align_by_element_width() -> None:
    assert cap.tma_align(4) == 4
    assert cap.tma_align(2) == 8
    with pytest.raises(ValueError):
        cap.tma_align(3)


# ---------------------------------------------------------------------------
# paged MQA logits boundaries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stride", [8, 16, 64, 1024])
def test_paged_mqa_bf16_aligned(stride: int) -> None:
    assert cap.paged_mqa_logits_ok(*BF16, stride, seq_len=100, page_size=64)


@pytest.mark.parametrize("stride", [1, 7, 9])
def test_paged_mqa_bf16_unaligned(stride: int) -> None:
    assert not cap.paged_mqa_logits_ok(*BF16, stride, seq_len=100, page_size=64)


def test_paged_mqa_empty_row_is_valid() -> None:
    assert cap.paged_mqa_logits_ok(*BF16, 1024, seq_len=0, page_size=64)


def test_paged_mqa_negative_seq_len_rejected() -> None:
    assert not cap.paged_mqa_logits_ok(*BF16, 1024, seq_len=-1, page_size=64)


def test_paged_mqa_zero_page_size_rejected() -> None:
    assert not cap.paged_mqa_logits_ok(*BF16, 1024, seq_len=10, page_size=0)


def test_paged_mqa_fp32_alignment() -> None:
    assert cap.paged_mqa_logits_ok(*FP32, 4, seq_len=10, page_size=64)
    assert not cap.paged_mqa_logits_ok(*FP32, 3, seq_len=10, page_size=64)


# ---------------------------------------------------------------------------
# BF16x2 packed histogram extraction
# ---------------------------------------------------------------------------


def _bf16_bits(value: float) -> int:
    import struct

    return struct.unpack("<H", struct.pack("<e", value))[0]


@pytest.mark.parametrize("value", [0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0])
def test_bf16x2_matches_scalar(value: float) -> None:
    bits = _bf16_bits(value)
    word = bits | (bits << 16)
    lo, hi = cap.bf16x2_to_uint8x2(word)
    assert lo == cap.bf16_to_uint8(bits)
    assert hi == cap.bf16_to_uint8(bits)


def test_bf16x2_low_half_first() -> None:
    lo_bits = _bf16_bits(1.0)
    hi_bits = _bf16_bits(-1.0)
    word = lo_bits | (hi_bits << 16)
    lo, hi = cap.bf16x2_to_uint8x2(word)
    assert lo == cap.bf16_to_uint8(lo_bits)
    assert hi == cap.bf16_to_uint8(hi_bits)


def test_bf16x2_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        cap.bf16x2_to_uint8x2(1 << 32)


def test_bf16_key_is_monotonic() -> None:
    values = [-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0]
    keys = [cap.bf16_to_uint8(_bf16_bits(v)) for v in values]
    assert keys == sorted(keys)


def test_fp32_to_uint8_matches_bf16_for_exact_values() -> None:
    for v in [0.0, 1.0, -1.0, 2.0, -2.0]:
        assert cap.fp32_to_uint8(v) == cap.bf16_to_uint8(_bf16_bits(v))


# ---------------------------------------------------------------------------
# selection consistency: BF16 vs FP32
# ---------------------------------------------------------------------------


def _bf16_round(value: float) -> float:
    import struct

    return struct.unpack("<e", struct.pack("<e", value))[0]


@pytest.mark.parametrize("k", [1, 4, 8])
@pytest.mark.parametrize("length", [16, 64, 256])
def test_bf16_and_fp32_select_same_values(k: int, length: int) -> None:
    # Values chosen so BF16 rounding is exact (powers of two and small ints).
    fp32_scores = [float((i * 37) % 101) for i in range(length)]
    bf16_scores = [_bf16_round(v) for v in fp32_scores]
    fp32_idx = cap.select_topk_indices(fp32_scores, k)
    bf16_idx = cap.select_topk_indices(bf16_scores, k)
    assert [fp32_scores[i] for i in fp32_idx] == [bf16_scores[i] for i in bf16_idx]


def test_select_topk_ties_by_index() -> None:
    scores = [1.0, 1.0, 1.0, 0.0]
    assert cap.select_topk_indices(scores, 2) == [0, 1]


def test_select_topk_k_larger_than_length() -> None:
    assert cap.select_topk_indices([3.0, 1.0], 5) == [0, 1]


def test_select_topk_negative_k_rejected() -> None:
    with pytest.raises(ValueError):
        cap.select_topk_indices([1.0], -1)


# ---------------------------------------------------------------------------
# indexer_topk backend resolution
# ---------------------------------------------------------------------------


def test_indexer_topk_resolution_declares_bf16() -> None:
    res = cap.resolve_topk_dtype("cooperative", *BF16, 1024, num_rows=4)
    assert res.dtype == "bf16"
    assert res.upcast is False


def test_indexer_topk_resolution_falls_back() -> None:
    res = cap.resolve_topk_dtype("cooperative", *BF16, 1023, num_rows=4)
    assert res.dtype == "fp32"
    assert res.upcast is True


def test_indexer_topk_auto_keeps_interface() -> None:
    # ``auto`` must remain a valid backend name for existing callers.
    res = cap.resolve_topk_dtype("auto", *FP32, 1024, num_rows=1)
    assert res.backend == "cooperative"
    assert res.dtype == "fp32"


def test_optional_int_passthrough() -> None:
    assert cap.optional_int(None) is None
    assert cap.optional_int(3) == 3
