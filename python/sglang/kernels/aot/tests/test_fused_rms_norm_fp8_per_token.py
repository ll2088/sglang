# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 SGLang Team
"""Fused (gated) RMSNorm + per-token FP8 quant vs norm followed by per-token quant.

Adapted from NVIDIA/TensorRT-LLM PR #21287 test coverage.
"""

import sys

import pytest
import torch

from sglang.kernels.ops.layernorm.fused_rms_norm_fp8_per_token import (
    add_rms_norm_fp8_per_token,
    rms_norm_gated_fp8_per_token,
)


def _per_token_quant(y: torch.Tensor):
    from sglang.kernels.ops.quantization.fp8_kernel import per_token_quant_fp8

    return per_token_quant_fp8(y)


def _assert_quant_matches(q, scale, y):
    # Must match quantizing the kernel's own normed output exactly.
    q_ref, scale_ref = _per_token_quant(y)
    torch.testing.assert_close(scale, scale_ref, rtol=0, atol=0)
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))


@pytest.mark.parametrize("num_tokens", [1, 50, 3000])
@pytest.mark.parametrize("use_gemma", [False, True])
@pytest.mark.parametrize("with_residual", [False, True])
def test_add_rms_norm_fp8_per_token(num_tokens, use_gemma, with_residual):
    torch.manual_seed(0)
    hidden, eps = 2560, 1e-6
    x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device="cuda") * 3
    residual = torch.randn_like(x) * 10 if with_residual else None
    weight = torch.randn(hidden, dtype=torch.bfloat16, device="cuda") * 0.2

    ref_in = x.float() + residual.float() if with_residual else x.float()
    gain = weight.float() + 1 if use_gemma else weight.float()
    ref = ref_in * torch.rsqrt(ref_in.pow(2).mean(-1, keepdim=True) + eps) * gain

    res_out = residual.clone() if with_residual else None
    y, q, scale = add_rms_norm_fp8_per_token(x, res_out, weight, eps, use_gemma)

    if with_residual:
        assert torch.equal(res_out, ref_in.to(torch.bfloat16))
    torch.testing.assert_close(y.float(), ref, rtol=1e-2, atol=1e-2)
    assert scale.shape == (num_tokens, 1) and scale.dtype == torch.float32
    _assert_quant_matches(q, scale, y)


@pytest.mark.parametrize("gate_activation", ["silu", "sigmoid"])
@pytest.mark.parametrize("num_tokens", [1, 50, 3000])
def test_rms_norm_gated_fp8_per_token(num_tokens, gate_activation):
    torch.manual_seed(0)
    heads, head_dim, eps = 32, 128, 1e-6
    x = torch.randn(num_tokens * heads, head_dim, dtype=torch.bfloat16, device="cuda")
    # z as a column slice of a wider per-token projection, as GDN passes it.
    z = torch.randn(
        num_tokens, heads * head_dim + 512, dtype=torch.bfloat16, device="cuda"
    )
    z = z[:, : heads * head_dim].view(num_tokens * heads, head_dim)
    weight = torch.randn(head_dim, dtype=torch.bfloat16, device="cuda") * 0.2

    xf = x.float()
    ref = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * weight.float()
    zf = z.float()
    ref = ref * (torch.sigmoid(zf) if gate_activation == "sigmoid" else zf * torch.sigmoid(zf))

    y, q, scale = rms_norm_gated_fp8_per_token(
        x, z, weight, eps, gate_activation=gate_activation
    )

    torch.testing.assert_close(y.float(), ref, rtol=1e-2, atol=1e-2)
    assert scale.shape == (num_tokens * heads, 1) and scale.dtype == torch.float32
    _assert_quant_matches(q, scale, y)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
