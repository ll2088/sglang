"""Triton implementation of fused per-tensor FP8 quantization.

Imported lazily from ``per_tensor_fp8_quant`` so that the public module stays
importable without torch/triton.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.per_tensor_fp8_quant import (
    FP8_MAX,
    MIN_DESCALE,
)

_BLOCK = 4096
_PARTS = 256


@triton.jit
def _partial_amax_kernel(
    X,
    Partial,
    N,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    start = pid * BLOCK
    acc = tl.zeros((BLOCK,), tl.float32)
    for step in range(tl.cdiv(N - start, PARTS * BLOCK)):
        offsets = start + step * PARTS * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(X + offsets, offsets < N, other=0.0).to(tl.float32)
        acc = tl.maximum(acc, tl.abs(x))
    tl.store(Partial + pid, tl.max(acc, 0))


@triton.jit
def _quant_kernel(
    X,
    Y,
    Partial,
    Scale,
    N,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
    FP8_MAX: tl.constexpr,
    MIN_DESCALE: tl.constexpr,
):
    pid = tl.program_id(0)
    start = pid * BLOCK
    if start < N or start == 0:
        offsets = start + tl.arange(0, BLOCK)
        x = tl.load(X + offsets, offsets < N, other=0.0).to(tl.float32)
        amax = tl.max(tl.load(Partial + tl.arange(0, PARTS)), 0)
        descale = tl.where(amax > 0, amax / FP8_MAX, MIN_DESCALE)
        y = tl.maximum(tl.minimum(x / descale, FP8_MAX), -FP8_MAX)
        tl.store(Y + offsets, y.to(Y.dtype.element_ty), offsets < N)
        if pid == 0:
            tl.store(Scale, descale)


def _quant_one(x: torch.Tensor):
    n = x.numel()
    partial = torch.empty(_PARTS, dtype=torch.float32, device=x.device)
    scale = torch.empty(1, dtype=torch.float32, device=x.device)
    out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    grid = (_PARTS,)
    _partial_amax_kernel[grid](x, partial, n, PARTS=_PARTS, BLOCK=_BLOCK)
    qgrid = (triton.cdiv(max(n, 1), _BLOCK),)
    _quant_kernel[qgrid](
        x,
        out,
        partial,
        scale,
        n,
        PARTS=_PARTS,
        BLOCK=_BLOCK,
        FP8_MAX=FP8_MAX,
        MIN_DESCALE=MIN_DESCALE,
    )
    return out, scale


def fused_per_tensor_fp8_quant(*tensors):
    """Quantize each input tensor with its own dynamic per-tensor scale."""
    outputs = []
    descales = []
    for x in tensors:
        out, scale = _quant_one(x.contiguous())
        outputs.append(out)
        descales.append(scale)
    return outputs, descales


__all__ = ["fused_per_tensor_fp8_quant"]
