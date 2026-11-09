"""Per-tensor FP8 quantization for attention inputs.

``fused_per_tensor_fp8_quant`` is the Triton entry point (registered in
``sglang.kernels.ops.attention``); ``reference_per_tensor_fp8_quant`` is a
pure-torch reference used by tests and by callers that must not import
triton. Both quantize each ``[tokens, heads, dim]`` input with its own amax
and FP32 descale, matching the upstream fused kernel's contract.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch

FP8_MAX = 448.0
MIN_DESCALE = 1e-6


def reference_per_tensor_fp8_quant(
    *inputs: torch.Tensor,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Quantize each input with its own amax/descale.

    Returns ``(outputs, descales)``; each descale is a 1-element FP32 tensor.
    Empty or all-zero inputs get ``MIN_DESCALE`` so downstream attention does
    not produce NaNs.
    """
    outputs: List[torch.Tensor] = []
    descales: List[torch.Tensor] = []
    for x in inputs:
        amax = x.abs().float().max() if x.numel() else x.new_zeros((), dtype=torch.float)
        descale = amax / FP8_MAX if amax > 0 else amax.new_tensor(MIN_DESCALE)
        q = (x.float() / descale).clamp(-FP8_MAX, FP8_MAX).to(fp8_dtype)
        outputs.append(q.contiguous())
        descales.append(descale.reshape(1))
    return outputs, descales


def fused_per_tensor_fp8_quant(
    *inputs: torch.Tensor,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Quantize up to three attention inputs with one fused Triton launch.

    Falls back to the torch reference when triton is unavailable so callers
    keep working on CPU-only or triton-less environments.
    """
    if not inputs:
        return [], []
    if len(inputs) > 3:
        raise ValueError(f"fused_per_tensor_fp8_quant takes at most 3 inputs, got {len(inputs)}")
    try:
        from sglang.kernels.ops.attention.per_tensor_fp8_quant_triton import (
            fused_per_tensor_fp8_quant as _triton_impl,
        )
    except ImportError:
        return reference_per_tensor_fp8_quant(*inputs, fp8_dtype=fp8_dtype)
    return _triton_impl(*inputs, fp8_dtype=fp8_dtype)


__all__ = [
    "FP8_MAX",
    "MIN_DESCALE",
    "fused_per_tensor_fp8_quant",
    "reference_per_tensor_fp8_quant",
]
