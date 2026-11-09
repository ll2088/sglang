"""Triton per-tensor FP8 quantizer (CUDA/ROCm only).

Kept in its own module so importing ``per_tensor_fp8_quant`` never requires
Triton; the wrapper falls back to the torch reference when this import fails.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import triton
import triton.language as tl

_FP8_MAX = 448.0
_MIN_DESCALE = 1e-6
_BLOCK = 4096


@triton.jit
def _amax_kernel(X, Out, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    acc = tl.zeros((BLOCK,), tl.float32)
    start = pid * BLOCK
    for step in range(tl.cdiv(N - start, tl.num_programs(0) * BLOCK)):
        offs = start + step * tl.num_programs(0) * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(X + offs, offs < N, other=0.0).to(tl.float32)
        acc = tl.maximum(acc, tl.abs(x))
    tl.store(Out + pid, tl.max(acc, 0))


@triton.jit
def _quant_kernel(X, Y, Amax, N, FP8_MAX: tl.constexpr, MIN_DESCALE: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    start = pid * BLOCK
    if start < N or start == 0:
        offs = start + tl.arange(0, BLOCK)
        x = tl.load(X + offs, offs < N, other=0.0).to(tl.float32)
        amax = tl.load(Amax)
        descale = tl.where(amax > 0, amax / FP8_MAX, MIN_DESCALE)
        y = tl.maximum(tl.minimum(x / descale, FP8_MAX), -FP8_MAX)
        tl.store(Y + offs, y.to(Y.dtype.element_ty), offs < N)


def _quant_one(x: torch.Tensor, fp8_dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    n = x.numel()
    x_flat = x.reshape(-1)
    y = torch.empty_like(x_flat, dtype=fp8_dtype)
    parts = max(1, min(256, triton.cdiv(n, _BLOCK) if n else 1))
    partial = torch.empty(parts, dtype=torch.float32, device=x.device)
    if n:
        _amax_kernel[(parts,)](x_flat, partial, n, BLOCK=_BLOCK)
    amax = partial.max() if n else x.new_zeros((), dtype=torch.float32)
    descale = (amax / _FP8_MAX) if amax > 0 else x.new_tensor(_MIN_DESCALE)
    grid = (max(1, triton.cdiv(n, _BLOCK)),)
    _quant_kernel[grid](
        x_flat, y, amax.reshape(1), n,
        FP8_MAX=_FP8_MAX, MIN_DESCALE=_MIN_DESCALE, BLOCK=_BLOCK,
    )
    return y.reshape(x.shape), descale.reshape(1).to(torch.float32)


def triton_per_tensor_fp8_quant(
    *inputs: torch.Tensor,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    out: Optional[Sequence[torch.Tensor]] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    outputs: List[torch.Tensor] = []
    descales: List[torch.Tensor] = []
    for i, x in enumerate(inputs):
        if out is not None and out[i] is not None:
            y, d = _quant_one(x, fp8_dtype)
            out[i].copy_(y)
            outputs.append(out[i])
        else:
            y, d = _quant_one(x, fp8_dtype)
            outputs.append(y)
        descales.append(d)
    return outputs, descales
