"""Per-tensor FP8 quantization helpers for MLA prefill.

``reference_per_tensor_fp8_quant`` is a pure-torch reference used by tests and
as a fallback when Triton is unavailable; ``fused_per_tensor_fp8_quant`` is the
Triton entry point registered in ``sglang.kernels.ops.attention``. Both take up
to three ``[tokens, heads, dim]`` inputs and return per-input FP8 tensors plus
FP32 descales, matching the upstream fused quantizer contract.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch

_FP8_MAX = 448.0
_MIN_DESCALE = 1e-6


def _descale_for(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x.new_tensor(_MIN_DESCALE, dtype=torch.float32)
    amax = x.detach().float().abs().max()
    if not torch.isfinite(amax) or amax <= 0:
        return x.new_tensor(_MIN_DESCALE, dtype=torch.float32)
    return (amax / _FP8_MAX).to(torch.float32)


def reference_per_tensor_fp8_quant(
    *inputs: torch.Tensor,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Quantize each input with its own amax/descale (torch reference)."""
    if not 1 <= len(inputs) <= 3:
        raise ValueError(f"expected 1..3 inputs, got {len(inputs)}")
    outputs: List[torch.Tensor] = []
    descales: List[torch.Tensor] = []
    for x in inputs:
        descale = _descale_for(x)
        q = (x.detach().float() / descale).clamp(-_FP8_MAX, _FP8_MAX)
        outputs.append(q.to(fp8_dtype).contiguous())
        descales.append(descale.reshape(1))
    return outputs, descales


def fused_per_tensor_fp8_quant(
    *inputs: torch.Tensor,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    out: Optional[Sequence[torch.Tensor]] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Fused per-tensor FP8 quantizer.

    Uses the Triton kernel when available and the inputs live on a CUDA/ROCm
    device; otherwise falls back to the torch reference so callers (and tests)
    work on CPU. The numerical contract is identical: one amax/descale per
    input, positive descale for empty or all-zero inputs.
    """
    if not 1 <= len(inputs) <= 3:
        raise ValueError(f"expected 1..3 inputs, got {len(inputs)}")
    if inputs[0].is_cuda:
        try:
            from sglang.kernels.ops.attention.per_tensor_fp8_quant_triton import (
                triton_per_tensor_fp8_quant,
            )

            return triton_per_tensor_fp8_quant(*inputs, fp8_dtype=fp8_dtype, out=out)
        except ImportError:
            pass
    return reference_per_tensor_fp8_quant(*inputs, fp8_dtype=fp8_dtype)
