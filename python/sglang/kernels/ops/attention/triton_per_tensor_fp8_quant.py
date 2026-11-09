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
triton_fused_qkv_quant.py, MIT license) via vllm-project/vllm#59124.
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
        heads, dim = SHAPE
        offsets = offsets.to(tl.int64)
        physical = (
            offsets // (heads * dim) * STRIDE[0]
            + offsets // dim % heads * STRIDE[1]
            + offsets % dim * STRIDE[2]
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
    Scale0,
    Scale1,
    Scale2,
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
            X0, Y0, Partial, Scale0, N0, SHAPES[0], STRIDES[0],
            PARTS, BLOCK, FP8_MAX, MIN_DESCALE,
        )
    elif idx == 1:
        _quant(
            X1, Y1, Partial + PARTS, Scale1, N1, SHAPES[1], STRIDES[1],
            PARTS, BLOCK, FP8_MAX, MIN_DESCALE,
        )
    else:
        _quant(
            X2, Y2, Partial + 2 * PARTS, Scale2, N2, SHAPES[2], STRIDES[2],
            PARTS, BLOCK, FP8_MAX, MIN_DESCALE,
        )


def _shape_stride(x: torch.Tensor) -> Tuple[Tuple[int, int], Tuple[int, int, int]]:
    assert x.dim() == 3, f"expected a 3D [tokens, heads, dim] tensor, got {x.shape}"
    assert x.stride(2) == 1, f"inner dim must be contiguous, got stride {x.stride()}"
    return (x.shape[1], x.shape[2]), (x.stride(0), x.stride(1), x.stride(2))


def fused_per_tensor_fp8_quant(
    *inputs: torch.Tensor,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    out: Optional[Sequence[Optional[torch.Tensor]]] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Dynamically quantize 1-3 attention inputs to FP8 with per-tensor descales.

    Each input is a ``[tokens, heads, dim]`` tensor whose inner dim is
    contiguous (strided views are fine). Returns ``(outputs, descales)`` where
    ``outputs[i]`` is a contiguous FP8 tensor shaped like ``inputs[i]`` and
    ``descales[i]`` is a 1-element FP32 tensor such that
    ``inputs[i] ~= outputs[i].float() * descales[i]``.

    Empty and all-zero inputs get ``_MIN_DESCALE`` so downstream attention
    kernels never divide by zero.
    """
    if not 1 <= len(inputs) <= 3:
        raise ValueError(f"expected 1-3 inputs, got {len(inputs)}")
    if fp8_dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise ValueError(f"unsupported fp8 dtype {fp8_dtype}")
    fp8_max = float(torch.finfo(fp8_dtype).max)

    device = inputs[0].device
    for x in inputs:
        assert x.device == device, "all inputs must live on the same device"
        assert x.dtype in (torch.bfloat16, torch.float16), (
            f"inputs must be bf16/fp16, got {x.dtype}"
        )

    shapes = [_shape_stride(x) for x in inputs]
    # Pad to three entries so the kernel signature is fixed; unused slots alias
    # input 0 and are never read (their N is 0 and their grid branch is masked).
    while len(inputs) < 3:
        inputs = inputs + (inputs[0],)
        shapes = shapes + [shapes[0]]

    num_inputs = len(inputs)
    if out is None:
        outs: List[torch.Tensor] = [
            torch.empty(x.shape, dtype=fp8_dtype, device=x.device)
            for x in inputs[:num_inputs]
        ]
    else:
        outs = []
        for i, x in enumerate(inputs[:num_inputs]):
            o = out[i] if i < len(out) else None
            if o is None:
                o = torch.empty(x.shape, dtype=fp8_dtype, device=x.device)
            assert o.shape == x.shape and o.dtype == fp8_dtype and o.is_contiguous(), (
                f"out[{i}] must be a contiguous {fp8_dtype} tensor of shape {tuple(x.shape)}"
            )
            outs.append(o)

    ns = [x.numel() for x in inputs]
    partial = torch.empty(3 * _PARTS, dtype=torch.float32, device=device)
    scales = [torch.empty(1, dtype=torch.float32, device=device) for _ in range(3)]

    _fused_amax_kernel[(_PARTS, 3)](
        inputs[0], inputs[1], inputs[2],
        partial,
        ns[0], ns[1], ns[2],
        SHAPES=tuple(shapes),
        STRIDES=tuple(s[1] for s in shapes),
        PARTS=_PARTS,
        BLOCK=_BLOCK,
        num_warps=8,
    )

    max_n = max(ns)
    grid = (max(1, triton.cdiv(max_n, _BLOCK)), 3)
    _fused_quant_kernel[grid](
        inputs[0], inputs[1], inputs[2],
        outs[0], outs[1], outs[2],
        partial,
        scales[0], scales[1], scales[2],
        ns[0], ns[1], ns[2],
        SHAPES=tuple(shapes),
        STRIDES=tuple(s[1] for s in shapes),
        PARTS=_PARTS,
        BLOCK=_BLOCK,
        FP8_MAX=fp8_max,
        MIN_DESCALE=_MIN_DESCALE,
        num_warps=8,
    )

    return outs[:num_inputs], scales[:num_inputs]
