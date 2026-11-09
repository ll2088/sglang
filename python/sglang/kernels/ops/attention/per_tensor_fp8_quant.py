"""Fused dynamic per-tensor FP8 quantization of up to three attention inputs.

Each ``[tokens, heads, dim]`` input (possibly strided, e.g. V as a view of the
``kv_b_proj`` output) gets its own amax and FP32 descale. A partial-amax pass
and a reduce-and-quantize pass cover all inputs in two launches; the kernel
boundary is the only global barrier, so no atomics or zero-initialization are
needed and the ops are CUDA-graph safe.

Triton is imported lazily so that importing this module (and the attention
kernel registry) never requires triton to be installed.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

_BLOCK = 4096
_PARTS = 256
# Descale for all-zero or empty inputs. A positive value keeps downstream
# attention kernels from producing NaNs.
_MIN_DESCALE = 1e-6


def _triton():
    from sglang.kernels.jit import triton_utils

    return triton_utils


def _build_kernels():
    """Compile the Triton kernels on first use (requires triton)."""
    tu = _triton()
    tl, triton = tu.tl, tu.triton

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
        X, Partial, N, SHAPE: tl.constexpr, STRIDE: tl.constexpr,
        PARTS: tl.constexpr, BLOCK: tl.constexpr,
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
        X0, X1, X2, Partial, N0, N1, N2,
        SHAPES: tl.constexpr, STRIDES: tl.constexpr,
        PARTS: tl.constexpr, BLOCK: tl.constexpr,
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
        X, Y, Partial, Scale, N, SHAPE: tl.constexpr, STRIDE: tl.constexpr,
        PARTS: tl.constexpr, BLOCK: tl.constexpr,
        FP8_MAX: tl.constexpr, MIN_DESCALE: tl.constexpr,
    ):
        start = tl.program_id(0) * BLOCK
        if start < N or start == 0:
            offsets = start + tl.arange(0, BLOCK)
            x = _load(X, offsets, N, SHAPE, STRIDE)
            amax = tl.max(tl.load(Partial + tl.arange(0, PARTS)), 0)
            descale = tl.maximum(amax / FP8_MAX, MIN_DESCALE)
            if start == 0:
                tl.store(Scale, descale)
            y = (x / descale).to(tl.float8e4nv)
            tl.store(Y + offsets, y, offsets < N)

    return _fused_amax_kernel, _quant


_KERNELS: Optional[Tuple] = None


def _kernels():
    global _KERNELS
    if _KERNELS is None:
        _KERNELS = _build_kernels()
    return _KERNELS


def fused_per_tensor_fp8_quant(
    *inputs: torch.Tensor,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Quantize 1-3 BF16/FP16 tensors to FP8 with per-tensor descales.

    Returns ``(outputs, descales)``; each descale is a 1-element FP32 tensor.
    """
    if not 1 <= len(inputs) <= 3:
        raise ValueError(f"expected 1-3 inputs, got {len(inputs)}")
    fused_amax, quant = _kernels()
    tu = _triton()
    triton = tu.triton
    fp8_max = float(torch.finfo(torch.float8_e4m3fn).max)
    device = inputs[0].device
    shapes, strides, ns = [], [], []
    for x in inputs:
        assert x.dim() == 3 and x.stride(-1) == 1, x.shape
        shapes.append((x.shape[1], x.shape[2]))
        strides.append((x.stride(0), x.stride(1), x.stride(2)))
        ns.append(x.numel())
    while len(inputs) < 3:
        inputs = inputs + (inputs[-1],)
        shapes.append(shapes[-1])
        strides.append(strides[-1])
        ns.append(0)
    partial = torch.empty(3 * _PARTS, dtype=torch.float32, device=device)
    fused_amax[(_PARTS, 3)](
        inputs[0], inputs[1], inputs[2], partial, ns[0], ns[1], ns[2],
        SHAPES=tuple(shapes), STRIDES=tuple(strides),
        PARTS=_PARTS, BLOCK=_BLOCK,
    )
    outputs, descales = [], []
    grid = (triton.cdiv(max(ns), _BLOCK),)
    for i, x in enumerate(inputs[: len(outputs) or 3]):
        y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        scale = torch.empty(1, dtype=torch.float32, device=device)
        quant[grid](
            x, y, partial + i * _PARTS, scale, ns[i],
            SHAPE=shapes[i], STRIDE=strides[i],
            PARTS=_PARTS, BLOCK=_BLOCK,
            FP8_MAX=fp8_max, MIN_DESCALE=_MIN_DESCALE,
        )
        outputs.append(y)
        descales.append(scale)
    return outputs, descales
