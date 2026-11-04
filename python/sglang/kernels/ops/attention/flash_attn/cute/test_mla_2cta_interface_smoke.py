# Copyright (c) 2025, Tri Dao.
"""Interface-level smoke test for dense 2CTA MLA head padding.

This test needs a CUDA SM100 device and the full sglang install, so it is
skipped on CPU-only runners. It verifies that the public interface entry point
accepts a 65-127 head dense MQA configuration and produces output matching a
reference implementation within tolerance.
"""

import importlib.util

import pytest


def _has_sglang():
    return importlib.util.find_spec("sglang") is not None


def _has_cuda_sm100():
    try:
        import torch
    except ImportError:
        return False
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 10


pytestmark = pytest.mark.skipif(
    not (_has_sglang() and _has_cuda_sm100()),
    reason="needs sglang and a CUDA SM100 device",
)


@pytest.mark.parametrize("nheads", [65, 96, 127])
def test_dense_2cta_padded_heads_forward_matches_reference(nheads):
    import torch

    from sglang.kernels.ops.attention.flash_attn.cute.interface import (
        flash_attn_func,
    )

    torch.manual_seed(0)
    device, dtype = "cuda", torch.bfloat16
    batch, seqlen, head_dim, head_dim_v = 2, 64, 64, 512
    q = torch.randn(batch, seqlen, nheads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, seqlen, 1, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, seqlen, 1, head_dim_v, device=device, dtype=dtype)

    out = flash_attn_func(q, k, v, causal=True)
    assert out.shape == (batch, seqlen, nheads, head_dim_v)
    assert torch.isfinite(out).all()

    # Reference: per-head softmax attention in fp32.
    qf = q.float()
    kf = k.float().expand(-1, -1, nheads, -1)
    vf = v.float().expand(-1, -1, nheads, -1)
    scores = qf @ kf.transpose(-1, -2) / (head_dim**0.5)
    mask = torch.triu(
        torch.ones(seqlen, seqlen, device=device, dtype=torch.bool), diagonal=1
    )
    scores = scores.masked_fill(mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    ref = probs @ vf
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)
