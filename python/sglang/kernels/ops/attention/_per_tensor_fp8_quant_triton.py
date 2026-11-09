"""Triton implementation of fused per-tensor FP8 quantization.

Imported lazily by ``per_tensor_fp8_quant.fused_per_tensor_fp8_quant`` so that
module import never requires triton.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import triton
import triton.language as tl

_BLOCK = 4096
_PARTS = 256
_MIN_DESCALE = 1e-6


@triton.jit
def _load(X, offsets, N, SHAPE: tl.constexpr, STRIDE: tl.constexpr):
    heads, dim = SHAPE
    offsets = offsets.to(tl.int64)
    physical = (
        offsets // (heads * dim) * STRIDE[0]
        + offsets // dim % heads * STRIDE[1]
        + offsets % dim * STRIDE[2]
    )
    return tl.load(X + physical, offsets < N, other=0.0).to(tl.float32)


@triton.jit
def _partial_amax(
    X,
    Partial,
    N,
    SHAPE: tl.constexpr,
    STRIDE: tl.constexpr,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    start = tl.program_id(0) * BLOCK
    acc = tl.zeros((BLOCK,), tl.float32)
    for step in range(tl.cdiv(N - start, PARTS * BLOCK)):
        offsets = start + step * PARTS * BLOCK + tl.arange(0, BLOCK)
        x = _load(X, offsets, N, SHAPE, STRIDE)
        acc = tl.maximum(acc, tl.abs(x))
    tl.store(Partial + tl.program_id(0), tl.max(acc, 0))


@triton.jit
def _fused_amax_kernel(
    X0,
    X1,
    X2,
    Partial,
    N0,
    N1,
    N2,
    SHAPES: tl.constexpr,
    STRIDES: tl.constexpr,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    idx = tl.program_id(1)
    if idx == 0:
        _partial_amax(X0, Partial, N0, SHAPES[0], STRIDES[0], PARTS, BLOCK)
    elif idx == 1:
        _partial_amax(X1, Partial + PARTS, N1, SHAPES[1], STRIDES[1], PARTS, BLOCK)
    else:
        _partial_amax(X2, Partial + 2 * PARTS, N2, SHAPES[2], STRIDES[2], PARTS, BLOCK)


@triton.jit
def _quant(
    X,
    Y,
    Partial,
    Scale,
    N,
    SHAPE: tl.constexpr,
    STRIDE: tl.constexpr,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
    FP8_MAX: tl.constexpr,
    MIN_DESCALE: tl.constexpr,
):
    start = tl.program_id(0) * BLOCK
    if start < N or start == 0:
        offsets = start + tl.arange(0, BLOCK)
        x = _load(X, offsets, N, SHAPE, STRIDE)
        amax = tl.max(tl.load(Partial + tl.arange(0, PARTS)), 0)
        descale = tl.where(amax > 0, amax / FP8_MAX, MIN_DESCALE)
        if start == 0:
            tl.store(Scale, descale)
        y = (x / descale).to(tl.float8e4nv)
        tl.store(Y + offsets, y, offsets < N)


@triton.jit
def _fused_quant_kernel(
    X0,
    X1,
    X2,
    Y0,
    Y1,
    Y2,
    Partial,
    Scale,
    N0,
    N1,
    N2,
    SHAPES: tl.constexpr,
    STRIDES: tl.constexpr,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
    FP8_MAX: tl.constexpr,
    MIN_DESCALE: tl.constexpr,
):
    idx = tl.program_id(1)
    if idx == 0:
        _quant(X0, Y0, Partial, Scale, N0, SHAPES[0], STRIDES[0], PARTS, BLOCK, FP8_MAX, MIN_DESCALE)
    elif idx == 1:
        _quant(X1, Y1, Partial + PARTS, Scale + 1, N1, SHAPES[1], STRIDES[1], PARTS, BLOCK, FP8_MAX, MIN_DESCALE)
    else:
        _quant(X2, Y2, Partial + 2 * PARTS, Scale + 2, N2, SHAPES[2], STRIDES[2], PARTS, BLOCK, FP8_MAX, MIN_DESCALE)


def fused_per_tensor_fp8_quant(
    *inputs: torch.Tensor,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    if not 1 <= len(inputs) <= 3:
        raise ValueError(f"expected 1..3 inputs, got {len(inputs)}")
    xs = list(inputs)
    while len(xs) < 3:
        xs.append(xs[0])
    shapes = tuple(tuple(x.shape[1:]) for x in xs)
    strides = tuple(tuple(x.stride()) for x in xs)
    ns = [x.numel() for x in xs]
    outs = [torch.empty_like(x, dtype=torch.float8_e4m3fn) for x in xs]
    partial = torch.empty(3 * _PARTS, dtype=torch.float32, device=xs[0].device)
    scale = torch.empty(3, dtype=torch.float32, device=xs[0].device)
    grid = (triton.cdiv(max(ns), _BLOCK), 3)
    _fused_amax_kernel[grid](
        xs[0], xs[1], xs[2], partial, ns[0], ns[1], ns[2],
        SHAPES=shapes, STRIDES=strides, PARTS=_PARTS, BLOCK=_BLOCK,
    )
    _fused_quant_kernel[grid](
        xs[0], xs[1], xs[2], outs[0], outs[1], outs[2], partial, scale,
        ns[0], ns[1], ns[2],
        SHAPES=shapes, STRIDES=strides, PARTS=_PARTS, BLOCK=_BLOCK,
        FP8_MAX=448.0, MIN_DESCALE=_MIN_DESCALE,
    )
    n = len(inputs)
    return outs[:n], [scale[i : i + 1] for i in range(n)]
