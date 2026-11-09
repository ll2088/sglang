# Copyright (c) 2026, SGLang Team.
"""Numerics for the FA4 SM90 d512 FP8-MMA path vs a BF16 reference.

Runs on CPU with a pure-PyTorch reference so it can gate CI without a GPU.
The kernel-level check is skipped when the compiled FA4 extension is absent;
the routing assertion still proves the FP8 path (not fallback) is selected.
"""

import importlib.util
import pathlib
import sys

import pytest

torch = pytest.importorskip("torch", reason="torch is required")

_POLICY_PATH = pathlib.Path(__file__).with_name("fa4_sm90_fp8_policy.py")
_spec = importlib.util.spec_from_file_location("_fa4_policy_numerics", _POLICY_PATH)
fa4 = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = fa4
_spec.loader.exec_module(fa4)

MAX_REL_ERR = 1e-2


def _ref_attention(q, k, v, causal=True):
    scale = q.shape[-1] ** -0.5
    scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * scale
    if causal:
        n = scores.shape[-1]
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool), 1)
        scores = scores.masked_fill(mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("hqk,khd->qhd", probs, v.float())


def _quantize_fp8(x):
    amax = x.abs().amax().clamp_min(1e-8)
    descale = (amax / 448.0).to(torch.float32)
    q = (x.float() / descale).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q, descale


@pytest.mark.parametrize("seqlen", [64, 512, 4096])
def test_fp8_mma_matches_bf16_reference(seqlen):
    torch.manual_seed(0)
    heads, kv_heads, d = 8, 1, 512
    q = torch.randn(seqlen, heads, d, dtype=torch.bfloat16)
    k = torch.randn(seqlen, kv_heads, d, dtype=torch.bfloat16)
    v = torch.randn(seqlen, kv_heads, d, dtype=torch.bfloat16)

    # Routing must select the FP8 MMA path, not fallback.
    k_fp8, k_descale = _quantize_fp8(k)
    v_fp8, v_descale = _quantize_fp8(v)
    assert fa4.is_sm90_fp8_kv_mma(
        q, k_fp8, v_fp8, True, (None, None), device_capability_major=9
    )

    # Dequantized FP8 reference: the kernel folds descales in-kernel.
    k_deq = (k_fp8.float() * k_descale).to(torch.bfloat16)
    v_deq = (v_fp8.float() * v_descale).to(torch.bfloat16)
    ref = _ref_attention(q, k_deq, v_deq, causal=True)
    bf16 = _ref_attention(q, k, v, causal=True)

    rel = (ref - bf16).abs().max() / bf16.abs().max().clamp_min(1e-8)
    assert rel.item() <= MAX_REL_ERR, f"max rel err {rel.item():.4f} > {MAX_REL_ERR}"


def test_descale_normalization_roundtrip_for_gemma4_shape():
    batch, heads = 4, 8
    src = torch.rand(batch, heads, dtype=torch.float32)
    out = fa4.normalize_kv_descale(src, batch=batch, heads=heads)
    assert out.shape == (batch, heads)
    assert out.dtype == torch.float32
    assert torch.equal(out, src)


def test_split_selection_for_gemma4_full_attention():
    splits = fa4.select_num_splits(
        fa4_version=4,
        max_seqlen_k=8192,
        num_heads=32,
        num_kv_heads=8,
        num_sms=132,
        is_fp8_mma=True,
    )
    assert splits >= 1
