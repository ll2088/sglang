"""Fused dynamic per-tensor FP8 quantization of up to three attention inputs.

Each ``[tokens, heads, dim]`` input (possibly strided, e.g. V as a view of the
``kv_b_proj`` output) gets its own amax and FP32 descale. A partial-amax pass
and a reduce-and-quantize pass cover all inputs in two launches; the kernel
boundary is the only global barrier, so no atomics or zero-initialization are
needed and the ops are CUDA-graph safe. Compile-time specialization does not
depend on the token counts (only the grid size does), so varying batch shapes
do not recompile.

One grid covers all the inputs in each launch, so a program can land past the
end of a smaller input. Such programs do no work and compute no address, and
every load and store is masked regardless, so no out-of-range pointer is formed
or dereferenced.

Adapted from the fused QKV quantizer in ROCm/ATOM (atom/model_ops/
triton_fused_qkv_quant.py, MIT license).
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
    # The grid is fixed at PARTS, so the trip count is per-program: it drops to
    # zero once this program's stripe starts past the end of this input.
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
    # One grid covers all inputs, so it is sized for the largest. Programs
    # past this input's end do nothing; program 0 still writes the descale,
    # which is what an empty input needs.
    start = tl.program_id(0) * BLOCK
    if start < N or start == 0:
        offsets = start + tl.arange(0, BLOCK)
        x = _load(X, offsets, N, SHAPE, STRIDE)
        amax = tl.max(tl.load(Partial + tl.arange(0, PARTS)), 0)
        descale = tl.maximum(amax / FP8_MAX, MIN_DESCALE)
        if start == 0:
            tl.store(Scale, descale)
        y = x / descale
        y = tl.clamp(y, -FP8_MAX, FP8_MAX)
        physical = (
            offsets.to(tl.int64) // (SHAPE[0] * SHAPE[1]) * STRIDE[0]
            + offsets.to(tl.int64) // SHAPE[1] % SHAPE[0] * STRIDE[1]
            + offsets.to(tl.int64) % SHAPE[1] * STRIDE[2]
        )
        tl.store(Y + physical, y.to(Y.dtype.element_ty), offsets < N)


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


def _fp8_max(dtype: torch.dtype) -> float:
    return float(torch.finfo(dtype).max)


def fused_per_tensor_fp8_quant(
    *inputs: torch.Tensor,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    out: Optional[Sequence[torch.Tensor]] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Dynamically quantize 1-3 ``[tokens, heads, dim]`` tensors per-tensor to FP8.

    Returns ``(outputs, descales)`` where ``outputs[i]`` is a contiguous FP8
    tensor with the same shape as ``inputs[i]`` and ``descales[i]`` is a
    one-element FP32 tensor such that ``inputs[i] ~= outputs[i] * descales[i]``.
    Empty inputs get a positive descale (``_MIN_DESCALE``) so downstream
    attention does not produce NaNs.
    """
    if not 1 <= len(inputs) <= 3:
        raise ValueError(f"expected 1-3 inputs, got {len(inputs)}")
    if fp8_dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise ValueError(f"unsupported fp8 dtype {fp8_dtype}")
    device = inputs[0].device
    for x in inputs:
        if x.dim() != 3:
            raise ValueError(f"expected 3D [tokens, heads, dim], got {tuple(x.shape)}")
        if x.device != device:
            raise ValueError("all inputs must be on the same device")
        if x.stride(-1) != 1:
            raise ValueError("inputs must have stride-1 inner dim")

    if out is None:
        outputs = [torch.empty_like(x, dtype=fp8_dtype) for x in inputs]
    else:
        outputs = list(out)
        if len(outputs) != len(inputs):
            raise ValueError("out must match the number of inputs")
        for x, y in zip(inputs, outputs):
            if y.shape != x.shape or y.dtype != fp8_dtype or y.stride(-1) != 1:
                raise ValueError("out tensors must match input shape/dtype/stride")

    descales = [
        torch.empty(1, dtype=torch.float32, device=device) for _ in inputs
    ]

    # Pad to three slots; unused slots alias the first input and are skipped by
    # the kernel via N == 0.
    xs = list(inputs) + [inputs[0]] * (3 - len(inputs))
    ys = list(outputs) + [outputs[0]] * (3 - len(outputs))
    ns = [x.numel() for x in inputs] + [0] * (3 - len(inputs))
    shapes = tuple((x.shape[1], x.shape[2]) for x in xs)
    strides = tuple((x.stride(0), x.stride(1), x.stride(2)) for x in xs)

    partial = torch.empty(3 * _PARTS, dtype=torch.float32, device=device)
    scale = torch.empty(3, dtype=torch.float32, device=device)

    _fused_amax_kernel[(_PARTS, 3)](
        xs[0], xs[1], xs[2], partial,
        ns[0], ns[1], ns[2],
        SHAPES=shapes, STRIDES=strides, PARTS=_PARTS, BLOCK=_BLOCK,
    )

    max_n = max(ns)
    grid = (triton.cdiv(max_n, _BLOCK) if max_n > 0 else 1, 3)
    _fused_quant_kernel[grid](
        xs[0], xs[1], xs[2], ys[0], ys[1], ys[2], partial, scale,
        ns[0], ns[1], ns[2],
        SHAPES=shapes, STRIDES=strides, PARTS=_PARTS, BLOCK=_BLOCK,
        FP8_MAX=_fp8_max(fp8_dtype), MIN_DESCALE=_MIN_DESCALE,
    )

    for i in range(len(inputs)):
        descales[i].copy_(scale[i : i + 1])
    return outputs, descales
