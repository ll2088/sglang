# Copyright (c) 2026, SGLang Team.
"""Routing tests for the FA4 SM90 d512 FP8-MMA path (no GPU required)."""

from unittest.mock import patch

import pytest
import torch

from sglang.kernels.ops.attention import flash_attention_v4 as fa4


class _FakeCuda:
    def __init__(self, major: int):
        self.major = major

    def get_device_capability(self, device=None):
        return (self.major, 0)


def _tensor(dtype, head_dim=512, heads=8):
    return torch.empty(1, heads, head_dim, dtype=dtype)


def _call(q, k, v, *, causal=True, window_size=(None, None)):
    return fa4._is_sm90_fp8_kv_mma(q, k, v, causal, window_size)


def _normalize_descale(descale, *, batch, heads, device="cpu"):
    """Normalize a K/V descale tensor to the (batch, heads) f32 layout.

    Accepts scalars, (heads,), (batch, heads) and (batch, heads, 1, 1)
    layouts so callers can pass either FA3-style or FA4-style descales.
    """
    if descale is None:
        return None
    t = torch.as_tensor(descale, dtype=torch.float32, device=device)
    if t.ndim == 0:
        return t.expand(batch, heads).contiguous()
    if t.ndim == 1:
        assert t.shape[0] == heads, f"expected {heads} heads, got {t.shape[0]}"
        return t.unsqueeze(0).expand(batch, heads).contiguous()
    if t.ndim == 2:
        assert t.shape == (batch, heads), f"expected {(batch, heads)}, got {tuple(t.shape)}"
        return t.contiguous()
    if t.ndim == 4:
        assert t.shape[0] == batch and t.shape[1] == heads
        assert t.shape[2] == 1 and t.shape[3] == 1
        return t.reshape(batch, heads).contiguous()
    raise ValueError(f"unsupported descale layout: {tuple(t.shape)}")


def _select_num_splits(
    *,
    fa_version: int,
    max_seqlen_k: int,
    num_heads: int,
    num_kv_heads: int,
    num_sms: int,
    is_fp8_mma: bool,
    cap: int = 32,
) -> int:
    """Mirror the FA4 full-attention split auto-selection policy.

    FA4 full-attention layers (including the SM90 d512 FP8-MMA path) are
    allowed to pick splits automatically without the CUDA-graph cap of 32;
    FA3 keeps the cap. Returns 0 for "auto" when the caller should let the
    kernel decide, otherwise a concrete split count.
    """
    if fa_version == 3:
        return cap
    if not is_fp8_mma:
        return 0
    if max_seqlen_k <= 0 or num_sms <= 0:
        return 1
    qhead_per_kvhead = max(1, num_heads // max(1, num_kv_heads))
    # One CTA per (kv head, split); aim for at least two KV tiles per split.
    tiles = max(1, max_seqlen_k // 128)
    target = max(1, num_sms // max(1, num_kv_heads * qhead_per_kvhead))
    return max(1, min(target, tiles))


@pytest.mark.parametrize("major", [9])
@pytest.mark.parametrize("q_dtype", [torch.float16, torch.bfloat16])
def test_sm90_d512_fp8_kv_routes_to_mma(major, q_dtype):
    q = _tensor(q_dtype)
    k = _tensor(torch.float8_e4m3fn)
    v = _tensor(torch.float8_e4m3fn)
    with patch.object(fa4.torch, "cuda", _FakeCuda(major)):
        assert _call(q, k, v) is True


def test_non_sm90_falls_back():
    q = _tensor(torch.bfloat16)
    k = _tensor(torch.float8_e4m3fn)
    v = _tensor(torch.float8_e4m3fn)
    with patch.object(fa4.torch, "cuda", _FakeCuda(10)):
        assert _call(q, k, v) is False


def test_non_d512_falls_back():
    q = _tensor(torch.bfloat16, head_dim=256)
    k = _tensor(torch.float8_e4m3fn, head_dim=256)
    v = _tensor(torch.float8_e4m3fn, head_dim=256)
    with patch.object(fa4.torch, "cuda", _FakeCuda(9)):
        assert _call(q, k, v) is False


def test_non_fp8_kv_falls_back():
    q = _tensor(torch.bfloat16)
    k = _tensor(torch.bfloat16)
    v = _tensor(torch.bfloat16)
    with patch.object(fa4.torch, "cuda", _FakeCuda(9)):
        assert _call(q, k, v) is False


def test_noncausal_full_attention_rejected():
    q = _tensor(torch.bfloat16)
    k = _tensor(torch.float8_e4m3fn)
    v = _tensor(torch.float8_e4m3fn)
    with patch.object(fa4.torch, "cuda", _FakeCuda(9)):
        with pytest.raises(NotImplementedError):
            _call(q, k, v, causal=False)


def test_noncausal_sliding_window_allowed():
    q = _tensor(torch.bfloat16)
    k = _tensor(torch.float8_e4m3fn)
    v = _tensor(torch.float8_e4m3fn)
    with patch.object(fa4.torch, "cuda", _FakeCuda(9)):
        assert _call(q, k, v, causal=False, window_size=(128, 0)) is True


@pytest.mark.parametrize(
    "layout,expected_shape",
    [
        (0.5, (2, 4)),
        (torch.ones(4), (2, 4)),
        (torch.ones(2, 4), (2, 4)),
        (torch.ones(2, 4, 1, 1), (2, 4)),
    ],
)
def test_descale_layout_normalization(layout, expected_shape):
    out = _normalize_descale(layout, batch=2, heads=4)
    assert out.shape == expected_shape
    assert out.dtype == torch.float32
    assert out.is_contiguous()


def test_descale_layout_rejects_bad_shape():
    with pytest.raises(ValueError):
        _normalize_descale(torch.ones(3, 5, 2), batch=2, heads=4)


@pytest.mark.parametrize(
    "fa_version,is_fp8_mma,expected",
    [
        (3, False, 32),
        (3, True, 32),
        (4, False, 0),
        (4, True, 4),
    ],
)
def test_split_auto_selection(fa_version, is_fp8_mma, expected):
    splits = _select_num_splits(
        fa_version=fa_version,
        max_seqlen_k=4096,
        num_heads=8,
        num_kv_heads=1,
        num_sms=32,
        is_fp8_mma=is_fp8_mma,
    )
    assert splits == expected


def test_split_auto_selection_short_seqlen():
    splits = _select_num_splits(
        fa_version=4,
        max_seqlen_k=64,
        num_heads=8,
        num_kv_heads=1,
        num_sms=32,
        is_fp8_mma=True,
    )
    assert splits == 1


def test_non_fp8_fallback_keeps_original_kernel():
    q = _tensor(torch.bfloat16)
    k = _tensor(torch.bfloat16)
    v = _tensor(torch.bfloat16)
    with patch.object(fa4.torch, "cuda", _FakeCuda(9)):
        assert _call(q, k, v) is False


def test_stability_repeated_routing():
    q = _tensor(torch.bfloat16)
    k = _tensor(torch.float8_e4m3fn)
    v = _tensor(torch.float8_e4m3fn)
    with patch.object(fa4.torch, "cuda", _FakeCuda(9)):
        results = {_call(q, k, v) for _ in range(3)}
    assert results == {True}
