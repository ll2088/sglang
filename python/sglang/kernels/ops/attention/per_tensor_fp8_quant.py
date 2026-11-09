"""Fused dynamic per-tensor FP8 quantization of up to three attention inputs.

Each ``[tokens, heads, dim]`` input (possibly strided, e.g. V as a view of the
``kv_b_proj`` output) gets its own amax and FP32 descale. A partial-amax pass
and a reduce-and-quantize pass cover all inputs in two launches; the kernel
boundary is the only global barrier, so no atomics or zero-initialization are
needed and the ops are CUDA-graph safe. Compile-time specialization does not
depend on the token counts (only the grid size does), so varying batch shapes
do not recompile.

Adapted from the fused QKV quantizer in ROCm/ATOM
(atom/model_ops/triton_fused_qkv_quant.py, MIT license) via vllm-project/vllm
PR #59124.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import triton
import triton.language as tl

_BLOCK = 4096
_PARTS = 256
# Descale for all-zero or empty inputs. A positive value keeps downstream
# attention kernels from producing NaNs.
_MIN_DESCALE = 1e-6

_FP8_MAX = {
    torch.float8_e4m3fn: 448.0,
    torch.float8_e5m2: 57344.0,
}


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
        descale = tl.maximum(amax / FP8_MAX, MIN_DESCALE)
        if start == 0:
            tl.store(Scale, descale)
        y = tl.clamp(x / descale, -FP8_MAX, FP8_MAX)
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
        _quant(
            X0, Y0, Partial, Scale, N0, SHAPES[0], STRIDES[0],
            PARTS, BLOCK, FP8_MAX, MIN_DESCALE,
        )
    elif idx == 1:
        _quant(
            X1, Y1, Partial + PARTS, Scale + 1, N1, SHAPES[1], STRIDES[1],
            PARTS, BLOCK, FP8_MAX, MIN_DESCALE,
        )
    else:
        _quant(
            X2, Y2, Partial + 2 * PARTS, Scale + 2, N2, SHAPES[2], STRIDES[2],
            PARTS, BLOCK, FP8_MAX, MIN_DESCALE,
        )


def _check_input(x: torch.Tensor, name: str) -> None:
    if x.dim() != 3:
        raise ValueError(f"{name} must be 3-D [tokens, heads, dim], got {tuple(x.shape)}")
    if x.stride(-1) != 1:
        raise ValueError(f"{name} must have a stride-1 inner dim")


def reference_per_tensor_fp8_quant(
    *inputs: torch.Tensor,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Torch reference for :func:`fused_per_tensor_fp8_quant`."""
    fp8_max = _FP8_MAX[fp8_dtype]
    outputs: List[torch.Tensor] = []
    descales: List[torch.Tensor] = []
    for x in inputs:
        amax = x.abs().float().max() if x.numel() else x.new_zeros((), dtype=torch.float)
        descale = amax / fp8_max if amax > 0 else amax.new_tensor(_MIN_DESCALE)
        y = (x.float() / descale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
        outputs.append(y)
        descales.append(descale.reshape(1))
    return outputs, descales


def fused_per_tensor_fp8_quant(
    *inputs: torch.Tensor,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    out: Optional[Sequence[Optional[torch.Tensor]]] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Dynamically quantize up to three ``[tokens, heads, dim]`` inputs to FP8.

    Returns ``(outputs, descales)`` where each output is contiguous FP8 and each
    descale is a 1-element FP32 tensor. Empty inputs get ``_MIN_DESCALE``.
    """
    if not 1 <= len(inputs) <= 3:
        raise ValueError(f"expected 1..3 inputs, got {len(inputs)}")
    if fp8_dtype not in _FP8_MAX:
        raise ValueError(f"unsupported fp8 dtype {fp8_dtype}")
    for i, x in enumerate(inputs):
        _check_input(x, f"inputs[{i}]")
    device = inputs[0].device
    for i, x in enumerate(inputs):
        if x.device != device:
            raise ValueError(f"inputs[{i}] is on {x.device}, expected {device}")

    outs: List[torch.Tensor] = []
    for i, x in enumerate(inputs):
        if out is not None and i < len(out) and out[i] is not None:
            y = out[i]
            if y.shape != x.shape or y.dtype != fp8_dtype:
                raise ValueError(f"out[{i}] must be {tuple(x.shape)} {fp8_dtype}")
        else:
            y = torch.empty(x.shape, dtype=fp8_dtype, device=device)
        outs.append(y)

    descales = [torch.empty(1, dtype=torch.float32, device=device) for _ in inputs]
    partial = torch.empty(3 * _PARTS, dtype=torch.float32, device=device)

    # Pad to three inputs with zero-size views so the kernel signature is fixed.
    xs = list(inputs) + [inputs[0][:0]] * (3 - len(inputs))
    ys = outs + [outs[0][:0]] * (3 - len(outs))
    shapes = tuple(tuple(x.shape[1:]) for x in xs)
    strides = tuple(tuple(x.stride()) for x in xs)
    ns = tuple(x.numel() for x in xs)

    _fused_amax_kernel[(_PARTS, 3)](
        xs[0], xs[1], xs[2], partial, *ns,
        SHAPES=shapes, STRIDES=strides, PARTS=_PARTS, BLOCK=_BLOCK,
    )
    max_n = max(ns)
    grid = (triton.cdiv(max_n, _BLOCK) or 1, 3)
    scale_buf = torch.stack(descales)
    _fused_quant_kernel[grid](
        xs[0], xs[1], xs[2], ys[0], ys[1], ys[2], partial, scale_buf, *ns,
        SHAPES=shapes, STRIDES=strides, PARTS=_PARTS, BLOCK=_BLOCK,
        FP8_MAX=_FP8_MAX[fp8_dtype], MIN_DESCALE=_MIN_DESCALE,
    )
    return outs, descales
