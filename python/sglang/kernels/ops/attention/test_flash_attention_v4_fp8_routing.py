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
