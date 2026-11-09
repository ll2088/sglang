"""Fused dynamic per-tensor FP8 quantization of up to three attention inputs.

Each ``[tokens, heads, dim]`` input (possibly strided, e.g. V as a view of the
``kv_b_proj`` output) gets its own amax and FP32 descale. A partial-amax pass
and a reduce-and-quantize pass cover all inputs in two launches; the kernel
boundary is the only global barrier, so no atomics or zero-initialization are
needed and the ops are CUDA-graph safe.

Adapted from the fused QKV quantizer in ROCm/ATOM (MIT license).

``triton`` is imported lazily so that importing this module (and the attention
kernel registry that references it) never requires triton to be installed.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

_BLOCK = 4096
_PARTS = 256
# Descale for all-zero or empty inputs. A positive value keeps downstream
# attention kernels from producing NaNs.
_MIN_DESCALE = 1e-6

_FP8_MAX = 448.0


def _require_triton():
    try:
        import triton  # noqa: F401
        import triton.language as tl  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "fused_per_tensor_fp8_quant requires triton; install triton or use "
            "the torch reference path"
        ) from exc


def _reference_quant(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure-torch per-tensor FP8 e4m3 quantization (reference / fallback)."""
    if x.numel() == 0:
        descale = x.new_tensor(_MIN_DESCALE, dtype=torch.float32)
    else:
        amax = x.abs().float().max()
        descale = (
            amax / _FP8_MAX if amax > 0 else x.new_tensor(_MIN_DESCALE, dtype=torch.float32)
        )
    x_fp8 = (x.float() / descale).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    return x_fp8, descale.reshape(1)


def fused_per_tensor_fp8_quant(
    *inputs: torch.Tensor,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Quantize up to three attention inputs to FP8 e4m3 with per-tensor descales.

    Returns ``(outputs, descales)``; each output is contiguous FP8 with the same
    shape as its input, and each descale is a 1-element FP32 tensor.
    """
    if not 1 <= len(inputs) <= 3:
        raise ValueError(f"expected 1..3 inputs, got {len(inputs)}")
    _require_triton()
    # The Triton fast path is only used on CUDA/ROCm devices; otherwise fall
    # back to the torch reference so the function stays usable in tests.
    if not all(x.is_cuda for x in inputs):
        outs, descales = [], []
        for x in inputs:
            x_fp8, descale = _reference_quant(x)
            outs.append(x_fp8.contiguous())
            descales.append(descale)
        return outs, descales
    from sglang.kernels.ops.attention._per_tensor_fp8_quant_triton import (
        fused_per_tensor_fp8_quant as _impl,
    )

    return _impl(*inputs)


__all__ = ["fused_per_tensor_fp8_quant"]
