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

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - triton is optional at import time
    triton = None
    tl = None
    _HAS_TRITON = False

_BLOCK = 4096
_PARTS = 256
# Descale for all-zero or empty inputs. A positive value keeps downstream
# attention kernels from producing NaNs.
_MIN_DESCALE = 1e-6


def _require_triton() -> None:
    if not _HAS_TRITON:
        raise ImportError(
            "per_tensor_fp8_quant requires triton; install triton to use it"
        )


if _HAS_TRITON:

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
            _partial_amax(
                X1, Partial + PARTS, N1, SHAPES[1], STRIDES[1], PARTS, BLOCK
            )
        else:
            _partial_amax(
                X2, Partial + 2 * PARTS, N2, SHAPES[2], STRIDES[2], PARTS, BLOCK
            )

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
            y = tl.minimum(tl.maximum(x / descale, -FP8_MAX), FP8_MAX)
            tl.store(Y + offsets, y.to(Y.dtype.element_ty), offsets < N)
            if start == 0:
                tl.store(Scale, descale)

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


def _prepare_inputs(
    inputs: Sequence[torch.Tensor],
) -> Tuple[List[torch.Tensor], List[Tuple[int, int]], List[Tuple[int, int, int]], List[int]]:
    tensors: List[torch.Tensor] = []
    shapes: List[Tuple[int, int]] = []
    strides: List[Tuple[int, int, int]] = []
    sizes: List[int] = []
    for x in inputs:
        if x.dim() != 3:
            raise ValueError(f"expected 3D input, got shape {tuple(x.shape)}")
        if x.stride(-1) != 1:
            raise ValueError("input inner dim must be contiguous")
        tensors.append(x)
        shapes.append((int(x.shape[1]), int(x.shape[2])))
        strides.append((int(x.stride(0)), int(x.stride(1)), int(x.stride(2))))
        sizes.append(int(x.numel()))
    return tensors, shapes, strides, sizes


def fused_per_tensor_fp8_quant(
    *inputs: torch.Tensor,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    out: Optional[Sequence[torch.Tensor]] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Quantize up to three ``[tokens, heads, dim]`` tensors per-tensor to FP8.

    Returns ``(outputs, descales)`` where each output has the input's shape and
    ``fp8_dtype``, and each descale is a 1-element FP32 tensor on the input's
    device. Empty inputs get ``_MIN_DESCALE``. Inputs must be 3D with a
    contiguous inner dim; strided outer dims are supported.
    """
    _require_triton()
    if not 1 <= len(inputs) <= 3:
        raise ValueError(f"expected 1..3 inputs, got {len(inputs)}")
    if fp8_dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise ValueError(f"unsupported fp8 dtype {fp8_dtype}")
    tensors, shapes, strides, sizes = _prepare_inputs(inputs)
    device = tensors[0].device
    if out is None:
        outputs = [
            torch.empty(x.shape, dtype=fp8_dtype, device=x.device) for x in tensors
        ]
    else:
        outputs = list(out)
        if len(outputs) != len(tensors):
            raise ValueError("out must match the number of inputs")
    descales = [torch.empty(1, dtype=torch.float32, device=device) for _ in tensors]
    # Pad to three inputs so the kernel signature is fixed.
    while len(tensors) < 3:
        tensors.append(tensors[0])
        outputs.append(outputs[0])
        descales.append(descales[0])
        shapes.append(shapes[0])
        strides.append(strides[0])
        sizes.append(0)
    partial = torch.empty(3 * _PARTS, dtype=torch.float32, device=device)
    scale = torch.empty(3, dtype=torch.float32, device=device)
    fp8_max = _fp8_max(fp8_dtype)
    grid_amax = (_PARTS, 3)
    _fused_amax_kernel[grid_amax](
        tensors[0], tensors[1], tensors[2], partial,
        sizes[0], sizes[1], sizes[2],
        SHAPES=tuple(shapes), STRIDES=tuple(strides),
        PARTS=_PARTS, BLOCK=_BLOCK,
    )
    max_n = max(sizes)
    grid_quant = (triton.cdiv(max_n, _BLOCK), 3)
    _fused_quant_kernel[grid_quant](
        tensors[0], tensors[1], tensors[2],
        outputs[0], outputs[1], outputs[2],
        partial, scale,
        sizes[0], sizes[1], sizes[2],
        SHAPES=tuple(shapes), STRIDES=tuple(strides),
        PARTS=_PARTS, BLOCK=_BLOCK,
        FP8_MAX=fp8_max, MIN_DESCALE=_MIN_DESCALE,
    )
    return outputs[: len(inputs)], [scale[i : i + 1] for i in range(len(inputs))]


def reference_per_tensor_fp8_quant(
    x: torch.Tensor, fp8_dtype: torch.dtype = torch.float8_e4m3fn
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Torch reference for a single input; used by tests and fallbacks."""
    fp8_max = _fp8_max(fp8_dtype)
    amax = x.abs().float().max() if x.numel() else x.new_zeros((), dtype=torch.float)
    descale = amax / fp8_max if amax > 0 else amax.new_tensor(_MIN_DESCALE)
    x_fp8 = (x.float() / descale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    return x_fp8, descale.reshape(1)
