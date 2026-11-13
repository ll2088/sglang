# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 SGLang Team
"""(Add +) or gated RMSNorm fused with dynamic per-token FP8 quantization.

Adapted from the fused norm + per-token FP8 quantize kernel introduced in
NVIDIA/TensorRT-LLM PR #21287. The quantization matches the existing
``vectorized_per_token_fp8_quant`` on the normed output, so the fused kernel
can be dropped into any norm -> per-token FP8 quantize -> GEMM chain without
changing numerics.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

FP8_E4M3_MAX = 448.0
FP8_MIN_SCALE = 1.0 / (FP8_E4M3_MAX * 512.0)


@triton.jit
def _quantize_fp8_per_token(y, MIN_SCALE: tl.constexpr):
    """y: fp32 values of one token (any shape). Returns (fp8 codes as fp32, scale)."""
    amax = tl.max(tl.abs(y))
    scale = tl.maximum(tl.math.div_rn(amax, 448.0), MIN_SCALE)
    q = tl.clamp(tl.math.div_rn(y, scale), -448.0, 448.0)
    return q, scale


@triton.jit
def _add_rms_norm_fp8_per_token_kernel(
    X,
    R,
    W,
    Y,
    Q,
    S,
    stride_x,
    stride_r,
    stride_y,
    stride_q,
    N,
    eps,
    BLOCK_N: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    WEIGHT_IS_DELTA: tl.constexpr,
    MIN_SCALE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    if HAS_RESIDUAL:
        x += tl.load(R + row * stride_r + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(R + row * stride_r + cols, x.to(R.dtype.element_ty), mask=mask)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / N + eps)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    if WEIGHT_IS_DELTA:
        w += 1.0
    y = (x * rstd * w).to(Y.dtype.element_ty)
    tl.store(Y + row * stride_y + cols, y, mask=mask)
    q, scale = _quantize_fp8_per_token(
        tl.where(mask, y.to(tl.float32), 0.0), MIN_SCALE
    )
    tl.store(Q + row * stride_q + cols, q.to(Q.dtype.element_ty), mask=mask)
    tl.store(S + row, scale)


@triton.jit
def _rms_norm_gated_fp8_per_token_kernel(
    X,
    Z,
    W,
    Y,
    Q,
    S,
    stride_x,
    stride_z,
    stride_y,
    stride_q,
    N,
    eps,
    BLOCK_N: tl.constexpr,
    GATE_IS_SIGMOID: tl.constexpr,
    MIN_SCALE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    z = tl.load(Z + row * stride_z + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / N + eps)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    if GATE_IS_SIGMOID:
        gate = tl.sigmoid(z)
    else:
        gate = z * tl.sigmoid(z)
    y = (x * rstd * w * gate).to(Y.dtype.element_ty)
    tl.store(Y + row * stride_y + cols, y, mask=mask)
    q, scale = _quantize_fp8_per_token(
        tl.where(mask, y.to(tl.float32), 0.0), MIN_SCALE
    )
    tl.store(Q + row * stride_q + cols, q.to(Q.dtype.element_ty), mask=mask)
    tl.store(S + row, scale)


def add_rms_norm_fp8_per_token(
    x: torch.Tensor,
    residual: Optional[torch.Tensor],
    weight: torch.Tensor,
    eps: float,
    weight_is_delta: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (y, fp8(y), fp32 scale [M, 1]) for y = rmsnorm(x [+ residual]).

    Updates ``residual`` in place when provided; ``weight_is_delta`` scales by
    (1 + weight) (gemma-style norm).
    """
    M, N = x.shape
    assert x.stride(-1) == 1 and (residual is None or residual.stride(-1) == 1)
    y = torch.empty_like(x)
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scale = torch.empty((M, 1), dtype=torch.float32, device=x.device)
    if M == 0:
        return y, q, scale
    block_n = triton.next_power_of_2(N)
    _add_rms_norm_fp8_per_token_kernel[(M,)](
        x,
        residual,
        weight,
        y,
        q,
        scale,
        x.stride(0),
        residual.stride(0) if residual is not None else 0,
        y.stride(0),
        q.stride(0),
        N,
        eps,
        BLOCK_N=block_n,
        HAS_RESIDUAL=residual is not None,
        WEIGHT_IS_DELTA=weight_is_delta,
        MIN_SCALE=FP8_MIN_SCALE,
    )
    return y, q, scale


def rms_norm_gated_fp8_per_token(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    gate_activation: str = "silu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (y, fp8(y), fp32 scale [M, 1]) for y = rmsnorm(x) * act(z).

    ``z`` may be a strided view (e.g. a column slice of a wider projection).
    """
    M, N = x.shape
    assert x.stride(-1) == 1 and z.stride(-1) == 1
    assert z.shape == x.shape
    assert gate_activation in ("silu", "sigmoid")
    y = torch.empty_like(x)
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scale = torch.empty((M, 1), dtype=torch.float32, device=x.device)
    if M == 0:
        return y, q, scale
    block_n = triton.next_power_of_2(N)
    _rms_norm_gated_fp8_per_token_kernel[(M,)](
        x,
        z,
        weight,
        y,
        q,
        scale,
        x.stride(0),
        z.stride(0),
        y.stride(0),
        q.stride(0),
        N,
        eps,
        BLOCK_N=block_n,
        GATE_IS_SIGMOID=(gate_activation == "sigmoid"),
        MIN_SCALE=FP8_MIN_SCALE,
    )
    return y, q, scale
