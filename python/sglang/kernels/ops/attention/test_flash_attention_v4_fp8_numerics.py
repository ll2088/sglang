# Copyright (c) 2026, SGLang Team.
"""CPU-only numerics check for the FA4 SM90 d512 FP8-MMA routing policy.

The real kernel needs an SM90 GPU and a compiled flash-attention extension,
so this test validates the parts that are pure host logic: the d512 routing
gate, descale layout normalization, and that a dequantized-FP8 reference
matches a BF16 reference within the acceptance tolerance. It also asserts
that the FP8 path is actually selected (not a silent fallback).
"""

import importlib.util
import pathlib
import sys
from unittest.mock import patch

import pytest
import torch

_POLICY_PATH = pathlib.Path(__file__).with_name("fa4_sm90_fp8_policy.py")
_POLICY_MODULE_NAME = "_fa4_sm90_fp8_policy_numerics_under_test"
_spec = importlib.util.spec_from_file_location(_POLICY_MODULE_NAME, _POLICY_PATH)
fa4 = importlib.util.module_from_spec(_spec)
sys.modules[_POLICY_MODULE_NAME] = fa4
try:
    _spec.loader.exec_module(fa4)
except Exception:  # pragma: no cover
    sys.modules.pop(_POLICY_MODULE_NAME, None)
    raise

MAX_REL_ERR = 1e-2


class _FakeCuda:
    def __init__(self, major: int):
        self.major = major

    def get_device_capability(self, device=None):
        return (self.major, 0)


def _patch_capability(major: int):
    # Patch the module-level helper instead of torch.cuda so the test does not
    # depend on torch.cuda being importable/mockable on a CPU-only runner.
    return patch.object(
        fa4, "_device_capability_major", lambda _=None: major
    )


def _gemma4_full_attention_shape():
    # Gemma-4 full attention: head_dim 512, GQA 8:1, small batch.
    return dict(batch=2, heads=8, kv_heads=1, head_dim=512, seqlen=64)


def _reference_attention(q, k, v, causal=True):
    # q: (b, h, s, d), k/v: (b, kv, s, d) -> (b, h, s, d)
    kv_heads = k.shape[1]
    heads = q.shape[1]
    group = heads // kv_heads
    k = k.repeat_interleave(group, dim=1)
    v = v.repeat_interleave(group, dim=1)
    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    if causal:
        s = scores.shape[-1]
        mask = torch.triu(torch.ones(s, s, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def _quantize_fp8(x):
    return x.to(torch.float8_e4m3fn)


def _dequantize_fp8(x, descale):
    return x.to(torch.float32) * descale


def test_gemma4_d512_fp8_mma_is_selected():
    shape = _gemma4_full_attention_shape()
    q = torch.empty(1, shape["heads"], shape["head_dim"], dtype=torch.bfloat16)
    k = torch.empty(1, shape["kv_heads"], shape["head_dim"], dtype=torch.float8_e4m3fn)
    v = torch.empty(1, shape["kv_heads"], shape["head_dim"], dtype=torch.float8_e4m3fn)
    with _patch_capability(9):
        assert fa4.is_sm90_fp8_kv_mma(q, k, v, True, (None, None)) is True


def test_gemma4_d512_fp8_mma_not_selected_off_sm90():
    shape = _gemma4_full_attention_shape()
    q = torch.empty(1, shape["heads"], shape["head_dim"], dtype=torch.bfloat16)
    k = torch.empty(1, shape["kv_heads"], shape["head_dim"], dtype=torch.float8_e4m3fn)
    v = torch.empty(1, shape["kv_heads"], shape["head_dim"], dtype=torch.float8_e4m3fn)
    with _patch_capability(10):
        assert fa4.is_sm90_fp8_kv_mma(q, k, v, True, (None, None)) is False


def test_fp8_dequant_reference_matches_bf16_within_tolerance():
    shape = _gemma4_full_attention_shape()
    torch.manual_seed(0)
    q = torch.randn(shape["batch"], shape["heads"], shape["seqlen"], shape["head_dim"])
    k = torch.randn(shape["batch"], shape["kv_heads"], shape["seqlen"], shape["head_dim"])
    v = torch.randn(shape["batch"], shape["kv_heads"], shape["seqlen"], shape["head_dim"])

    ref = _reference_attention(q, k, v, causal=True)

    q_d = fa4.normalize_kv_descale(1.0, batch=shape["batch"], heads=shape["heads"])
    k_d = fa4.normalize_kv_descale(1.0, batch=shape["batch"], heads=shape["kv_heads"])
    v_d = fa4.normalize_kv_descale(1.0, batch=shape["batch"], heads=shape["kv_heads"])
    assert q_d == 1.0 and k_d == 1.0 and v_d == 1.0

    q_fp8 = _quantize_fp8(q)
    k_fp8 = _quantize_fp8(k)
    v_fp8 = _quantize_fp8(v)
    q_deq = _dequantize_fp8(q_fp8, q_d)
    k_deq = _dequantize_fp8(k_fp8, k_d)
    v_deq = _dequantize_fp8(v_fp8, v_d)

    out = _reference_attention(q_deq, k_deq, v_deq, causal=True)
    denom = ref.abs().clamp_min(1e-6)
    max_rel_err = ((out - ref).abs() / denom).max().item()
    assert max_rel_err <= MAX_REL_ERR, f"max relative error {max_rel_err} > {MAX_REL_ERR}"


def test_fp8_path_is_not_a_fallback():
    # If routing silently fell back, the FP8 gate would return False for the
    # exact Gemma-4 full-attention shape. Assert the opposite.
    shape = _gemma4_full_attention_shape()
    q = torch.empty(1, shape["heads"], shape["head_dim"], dtype=torch.bfloat16)
    k = torch.empty(1, shape["kv_heads"], shape["head_dim"], dtype=torch.float8_e4m3fn)
    v = torch.empty(1, shape["kv_heads"], shape["head_dim"], dtype=torch.float8_e4m3fn)
    with _patch_capability(9):
        assert fa4.is_sm90_fp8_kv_mma(q, k, v, True, (None, None)) is True


def test_split_selection_for_gemma4_shape():
    splits = fa4.select_num_splits(
        fa_version=4,
        max_seqlen_k=4096,
        num_heads=8,
        num_kv_heads=1,
        num_sms=132,
        is_fp8_mma=True,
    )
    assert splits >= 1
    assert splits <= fa4.SM90_FP8_MMA_MAX_SPLITS
    assert splits <= 4096 // fa4.SM90_FP8_MMA_TILE_N


@pytest.mark.parametrize(
    "layout",
    [
        0.5,
        torch.ones(1),
        torch.ones(4),
        torch.ones(1, 4),
        torch.ones(2, 4),
        torch.ones(1, 1, 1, 1),
        torch.ones(2, 4, 1, 1),
    ],
)
def test_descale_layouts_are_normalized_for_gemma4(layout):
    out = fa4.normalize_kv_descale(layout, batch=2, heads=4)
    assert out.shape == (2, 4)
    assert out.dtype == torch.float32
    assert out.is_contiguous()


def test_descale_rejects_mismatched_layout():
    with pytest.raises(ValueError):
        fa4.normalize_kv_descale(torch.ones(3, 4), batch=2, heads=4)
    with pytest.raises(ValueError):
        fa4.normalize_kv_descale(torch.ones(3), batch=2, heads=4)


def test_noncausal_full_attention_is_rejected_on_sm90():
    shape = _gemma4_full_attention_shape()
    q = torch.empty(1, shape["heads"], shape["head_dim"], dtype=torch.bfloat16)
    k = torch.empty(1, shape["kv_heads"], shape["head_dim"], dtype=torch.float8_e4m3fn)
    v = torch.empty(1, shape["kv_heads"], shape["head_dim"], dtype=torch.float8_e4m3fn)
    with _patch_capability(9):
        with pytest.raises(NotImplementedError):
            fa4.is_sm90_fp8_kv_mma(q, k, v, False, (None, None))
        # A sliding window makes the noncausal call legal.
        assert fa4.is_sm90_fp8_kv_mma(q, k, v, False, (128, 0)) is True


def test_non_d512_or_non_fp8_falls_back():
    q = torch.empty(1, 8, 256, dtype=torch.bfloat16)
    k = torch.empty(1, 1, 256, dtype=torch.float8_e4m3fn)
    v = torch.empty(1, 1, 256, dtype=torch.float8_e4m3fn)
    with _patch_capability(9):
        assert fa4.is_sm90_fp8_kv_mma(q, k, v, True, (None, None)) is False
    q = torch.empty(1, 8, 512, dtype=torch.bfloat16)
    k = torch.empty(1, 1, 512, dtype=torch.bfloat16)
    v = torch.empty(1, 1, 512, dtype=torch.bfloat16)
    with _patch_capability(9):
        assert fa4.is_sm90_fp8_kv_mma(q, k, v, True, (None, None)) is False


def test_split_selection_scales_with_seqlen():
    short = fa4.select_num_splits(
        fa_version=4, max_seqlen_k=256, num_heads=8, num_kv_heads=1,
        num_sms=132, is_fp8_mma=True,
    )
    long = fa4.select_num_splits(
        fa_version=4, max_seqlen_k=16384, num_heads=8, num_kv_heads=1,
        num_sms=132, is_fp8_mma=True,
    )
    assert 1 <= short <= long <= fa4.SM90_FP8_MMA_MAX_SPLITS
    assert fa4.select_num_splits(
        fa_version=3, max_seqlen_k=16384, num_heads=8, num_kv_heads=1,
        num_sms=132, is_fp8_mma=True,
    ) == 32
    assert fa4.select_num_splits(
        fa_version=4, max_seqlen_k=16384, num_heads=8, num_kv_heads=1,
        num_sms=132, is_fp8_mma=False,
    ) == 0
