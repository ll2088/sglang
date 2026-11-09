"""Fused dynamic per-tensor FP8 quantization of up to three attention inputs.

Each ``[tokens, heads, dim]`` input (possibly strided, e.g. V as a view of the
``kv_b_proj`` output) gets its own amax and FP32 descale. A partial-amax pass
and a reduce-and-quantize pass cover all inputs in two launches; the kernel
boundary is the only global barrier, so no atomics or zero-initialization are
needed and the ops are CUDA-graph safe. Compile-time specialization does not
depend on the token counts (only the grid size does), so varying batch shapes
do not recompile.

Adapted from the fused QKV quantizer in ROCm/ATOM (MIT license) via
vllm-project/vllm#59124.
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

_FP8_DTYPE_MAP = {
    torch.float8_e4m3fn: tl.float8e4nv,
    torch.float8_e5m2: tl.float8e5,
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
        y = (x / descale).to(tl.float8e4nv if FP8_MAX == 448.0 else tl.float8e5)
        heads, dim = SHAPE
        offsets = offsets.to(tl.int64)
        physical = (
            offsets // (heads * dim) * STRIDE[0]
            + offsets // dim % heads * STRIDE[1]
            + offsets % dim * STRIDE[2]
        )
        tl.store(Y + physical, y, offsets < N)


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


def _shape_stride(x: torch.Tensor) -> Tuple[Tuple[int, int], Tuple[int, int, int]]:
    assert x.dim() == 3, f"expected 3D input, got {tuple(x.shape)}"
    assert x.stride(-1) == 1, "inner dim must be contiguous"
    return (x.shape[1], x.shape[2]), (x.stride(0), x.stride(1), x.stride(2))


def fused_per_tensor_fp8_quant(
    *inputs: torch.Tensor,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    out: Optional[Sequence[Optional[torch.Tensor]]] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Dynamically quantize 1-3 ``[tokens, heads, dim]`` tensors per-tensor.

    Returns ``(outputs, descales)`` where each output is a contiguous FP8 tensor
    with the same shape as its input and each descale is a 1-element FP32 tensor
    (``x_fp8 = x / descale``). Empty inputs get ``_MIN_DESCALE``.
    """
    if not 1 <= len(inputs) <= 3:
        raise ValueError(f"expected 1-3 inputs, got {len(inputs)}")
    if fp8_dtype not in _FP8_DTYPE_MAP:
        raise ValueError(f"unsupported fp8 dtype {fp8_dtype}")
    device = inputs[0].device
    for x in inputs:
        assert x.device == device, "all inputs must be on the same device"
        assert x.dtype in (torch.bfloat16, torch.float16), x.dtype

    shapes = [_shape_stride(x) for x in inputs]
    # Pad to three slots so the kernel signature is fixed.
    while len(inputs) < 3:
        inputs = inputs + (inputs[-1],)
        shapes = shapes + (shapes[-1],)

    outputs: List[torch.Tensor] = []
    for i, x in enumerate(inputs):
        if out is not None and i < len(out) and out[i] is not None:
            outputs.append(out[i])
        else:
            outputs.append(torch.empty(x.shape, dtype=fp8_dtype, device=device))

    partial = torch.empty(3 * _PARTS, dtype=torch.float32, device=device)
    descales = torch.empty(3, dtype=torch.float32, device=device)
    ns = [x.numel() for x in inputs]
    max_n = max(ns)
    grid_amax = (min(_PARTS, triton.cdiv(max_n, _BLOCK)), 3)
    grid_quant = (triton.cdiv(max_n, _BLOCK), 3)

    fp8_max = float(torch.finfo(fp8_dtype).max)
    _fused_amax_kernel[grid_amax](
        inputs[0], inputs[1], inputs[2], partial,
        ns[0], ns[1], ns[2],
        SHAPES=[s for s, _ in shapes],
        STRIDES=[st for _, st in shapes],
        PARTS=_PARTS, BLOCK=_BLOCK,
    )
    _fused_quant_kernel[grid_quant](
        inputs[0], inputs[1], inputs[2],
        outputs[0], outputs[1], outputs[2],
        partial, descales,
        ns[0], ns[1], ns[2],
        SHAPES=[s for s, _ in shapes],
        STRIDES=[st for _, st in shapes],
        PARTS=_PARTS, BLOCK=_BLOCK,
        FP8_MAX=fp8_max, MIN_DESCALE=_MIN_DESCALE,
    )
    return outputs, [descales[i : i + 1] for i in range(3)]
