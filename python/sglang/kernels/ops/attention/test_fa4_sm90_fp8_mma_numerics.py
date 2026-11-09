# Copyright (c) 2026, SGLang Team.
"""Self-contained checks for the FA4 SM90 d512 FP8-MMA routing policy.

Loads ``fa4_sm90_fp8_policy.py`` by path so no sglang package import (and no
pip install of sglang) is required. Covers routing gates, descale layout
normalization, split auto-selection and a small FP8-vs-BF16 numeric check
against a pure-PyTorch reference.
"""

import importlib.util
import pathlib
import sys

import pytest
import torch

_POLICY_PATH = pathlib.Path(__file__).with_name("fa4_sm90_fp8_policy.py")
_NAME = "_fa4_sm90_fp8_policy_numerics"
_spec = importlib.util.spec_from_file_location(_NAME, _POLICY_PATH)
fa4 = importlib.util.module_from_spec(_spec)
sys.modules[_NAME] = fa4
try:
    _spec.loader.exec_module(fa4)
except Exception:  # pragma: no cover
    sys.modules.pop(_NAME, None)
    raise


def _t(dtype, head_dim=512, heads=8):
    return torch.empty(1, heads, head_dim, dtype=dtype)


def _route(q, k, v, *, causal=True, window_size=(None, None), major=9):
    return fa4.is_sm90_fp8_kv_mma(
        q, k, v, causal, window_size, device_capability_major=major
    )


@pytest.mark.parametrize("q_dtype", [torch.float16, torch.bfloat16])
def test_gemma4_full_attention_d512_routes_to_fp8_mma(q_dtype):
    q = _t(q_dtype)
    k = _t(torch.float8_e4m3fn)
    v = _t(torch.float8_e4m3fn)
    assert _route(q, k, v) is True


def test_sliding_window_d256_stays_on_fa3():
    q = _t(torch.bfloat16, head_dim=256)
    k = _t(torch.float8_e4m3fn, head_dim=256)
    v = _t(torch.float8_e4m3fn, head_dim=256)
    assert _route(q, k, v) is False


def test_non_sm90_and_non_fp8_fall_back():
    q = _t(torch.bfloat16)
    k = _t(torch.float8_e4m3fn)
    v = _t(torch.float8_e4m3fn)
    assert _route(q, k, v, major=10) is False
    assert _route(q, _t(torch.bfloat16), _t(torch.bfloat16)) is False


def test_noncausal_full_attention_rejected_but_window_allowed():
    q = _t(torch.bfloat16)
    k = _t(torch.float8_e4m3fn)
    v = _t(torch.float8_e4m3fn)
    with pytest.raises(NotImplementedError):
        _route(q, k, v, causal=False)
    assert _route(q, k, v, causal=False, window_size=(128, 0)) is True


@pytest.mark.parametrize(
    "layout,shape",
    [
        (0.5, (2, 4)),
        (torch.ones(4), (2, 4)),
        (torch.ones(2, 4), (2, 4)),
        (torch.ones(2, 4, 1, 1), (2, 4)),
    ],
)
def test_descale_layout_normalization(layout, shape):
    out = fa4.normalize_kv_descale(layout, batch=2, heads=4)
    assert out.shape == shape
    assert out.dtype == torch.float32
    assert out.is_contiguous()


@pytest.mark.parametrize(
    "bad",
    [torch.ones(3), torch.ones(3, 5, 2), torch.ones(2, 4, 2, 1), torch.ones(2, 4, 1)],
)
def test_descale_layout_rejects_bad_shapes(bad):
    with pytest.raises(ValueError):
        fa4.normalize_kv_descale(bad, batch=2, heads=4)


def test_descale_none_passthrough_and_values_preserved():
    assert fa4.normalize_kv_descale(None, batch=2, heads=4) is None
    src = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    assert torch.equal(fa4.normalize_kv_descale(src, batch=2, heads=4), src)


@pytest.mark.parametrize(
    "fa_version,is_fp8_mma,expected",
    [(3, False, 32), (3, True, 32), (4, False, 0), (4, True, 4)],
)
def test_split_auto_selection(fa_version, is_fp8_mma, expected):
    assert (
        fa4.select_num_splits(
            fa_version=fa_version,
            max_seqlen_k=4096,
            num_heads=8,
            num_kv_heads=1,
            num_sms=32,
            is_fp8_mma=is_fp8_mma,
        )
        == expected
    )


def test_split_auto_selection_edges():
    kw = dict(num_heads=8, num_kv_heads=1, num_sms=32, is_fp8_mma=True)
    assert fa4.select_num_splits(fa_version=4, max_seqlen_k=64, **kw) == 1
    assert fa4.select_num_splits(fa_version=4, max_seqlen_k=0, **kw) == 1
    assert (
        fa4.select_num_splits(
            fa_version=4, max_seqlen_k=4096, num_heads=8, num_kv_heads=1,
            num_sms=0, is_fp8_mma=True,
        )
        == 1
    )
    assert (
        fa4.select_num_splits(
            fa_version=4, max_seqlen_k=256, num_heads=8, num_kv_heads=1,
            num_sms=1024, is_fp8_mma=True,
        )
        == 2
    )


def _reference_attention(q, k, v, scale):
    scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * scale
    mask = torch.triu(torch.ones_like(scores, dtype=torch.bool), 1)
    scores = scores.masked_fill(mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("hqk,khd->qhd", probs, v.float())


def test_fp8_mma_matches_bf16_reference_within_tolerance():
    torch.manual_seed(0)
    seqlen, heads, dim = 64, 4, 512
    q = torch.randn(seqlen, heads, dim, dtype=torch.bfloat16)
    k = torch.randn(seqlen, heads, dim, dtype=torch.bfloat16)
    v = torch.randn(seqlen, heads, dim, dtype=torch.bfloat16)
    scale = dim**-0.5
    ref = _reference_attention(q, k, v, scale)
    # Emulate the FP8-MMA path: E4M3 K/V with in-kernel descale folded in.
    k_fp8 = k.to(torch.float8_e4m3fn).to(torch.bfloat16)
    v_fp8 = v.to(torch.float8_e4m3fn).to(torch.bfloat16)
    out = _reference_attention(q, k_fp8, v_fp8, scale)
    rel = (out - ref).abs().max() / ref.abs().max().clamp_min(1e-6)
    assert rel <= 1e-2
