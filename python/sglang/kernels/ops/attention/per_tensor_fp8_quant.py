"""Per-tensor FP8 quantization helpers for MLA prefill.

This module is intentionally free of torch/triton imports at module scope so
that it can be imported (and its tests collected) on CPU-only or triton-less
environments. The actual GPU kernel lives in
``_per_tensor_fp8_quant_triton`` and is imported lazily.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

FP8_MAX = 448.0
MIN_DESCALE = 1e-6


def reference_per_tensor_fp8_quant(
    values: Sequence[float],
) -> Tuple[List[float], float]:
    """Pure-Python reference for dynamic per-tensor FP8 quantization.

    Returns ``(quantized, descale)`` where ``quantized`` is clamped to
    ``[-FP8_MAX, FP8_MAX]`` and ``descale`` is ``amax / FP8_MAX`` (or
    ``MIN_DESCALE`` for all-zero / empty inputs).
    """
    amax = max((abs(v) for v in values), default=0.0)
    descale = amax / FP8_MAX if amax > 0 else MIN_DESCALE
    quantized = [max(-FP8_MAX, min(FP8_MAX, v / descale)) for v in values]
    return quantized, descale


def fused_per_tensor_fp8_quant(*tensors):
    """Lazy entry point for the fused Triton quantizer.

    Importing this function does not import torch or triton; the heavy module
    is only loaded when the function is actually called on a GPU host.
    """
    from sglang.kernels.ops.attention._per_tensor_fp8_quant_triton import (
        fused_per_tensor_fp8_quant as _impl,
    )

    return _impl(*tensors)


__all__ = [
    "FP8_MAX",
    "MIN_DESCALE",
    "fused_per_tensor_fp8_quant",
    "reference_per_tensor_fp8_quant",
]
