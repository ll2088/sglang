"""Pure-torch FP4 fake quantization for DeepSeek-V4.1.

Indexer values use per-32 UE8M0 scales; compressed KV uses per-16 E4M3 scales.
Both paths round to the E2M1 grid with ties to even.
"""

from contextlib import contextmanager

import torch

FP8_MAX = 448.0
FP4_MAX = 6.0
FP8_BLOCK_SIZE = 32
FP4_BLOCK_SIZE = 32
FP4_AMAX_FLOOR = 6 * 2.0**-126


def ceil_pow2(x: torch.Tensor) -> torch.Tensor:
    """2 ** ceil(log2(x)) for positive fp32 x, computed on the IEEE bits so the
    result is exact at powers of two."""
    bits = x.contiguous().view(torch.int32)
    exponent = ((bits >> 23) & 0xFF) - 127
    has_mantissa = (bits & 0x7FFFFF) != 0
    exponent = exponent + has_mantissa.to(torch.int32)
    return ((exponent + 127) << 23).view(torch.float32)


def block_scale(x: torch.Tensor, block_size: int, fmax: float, amax_floor: float):
    """Per-block ue8m0 scale, as fp32 powers of two, shape [..., N // block_size]."""
    amax = x.float().unflatten(-1, (-1, block_size)).abs().amax(dim=-1)
    amax = amax.clamp_min(amax_floor)
    # The kernel multiplies by the fp32 reciprocal rather than dividing. A Python
    # scalar keeps this free of host tensors, so it can run under CUDA graph capture.
    return ceil_pow2(amax * (1.0 / fmax))


def round_fp4(x: torch.Tensor) -> torch.Tensor:
    """Round fp32 values in [-6, 6] onto the e2m1 grid with round-to-nearest-even."""
    magnitude = x.abs()
    step = torch.where(magnitude < 2.0, 0.5, torch.where(magnitude < 4.0, 1.0, 2.0))
    return torch.round(magnitude / step) * step * torch.sign(x)


def fake_quant_fp4(x: torch.Tensor, block_size: int = FP4_BLOCK_SIZE) -> torch.Tensor:
    """Quantize to fp4 (per-block ue8m0 scale) and back, in x's dtype."""
    scale = block_scale(x, block_size, FP4_MAX, FP4_AMAX_FLOOR)
    scaled = x.float().unflatten(-1, (-1, block_size)) / scale.unsqueeze(-1)
    deq = round_fp4(scaled.clamp(-FP4_MAX, FP4_MAX)) * scale.unsqueeze(-1)
    return deq.flatten(-2).to(x.dtype)


def fake_quant_compressed_kv(x: torch.Tensor) -> torch.Tensor:
    """FP4 round-trip with one E4M3FN scale per 16 compressed-KV elements.

    Round amax / 6 to E4M3 with ties to even, clamping the scale to its
    positive finite range [2**-9, 448]. Zero blocks remain zero.
    """
    blocks = x.float().unflatten(-1, (-1, 16))
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    scale = (amax * (1.0 / FP4_MAX)).clamp(min=2**-9, max=FP8_MAX)
    scale = scale.to(torch.float8_e4m3fn).float()
    scaled = (blocks / scale).clamp(-FP4_MAX, FP4_MAX)
    deq = round_fp4(scaled) * scale
    return deq.flatten(-2).to(x.dtype)


@contextmanager
def replace_function(obj, name: str, new):
    """Temporarily replace ``obj.name`` with ``new``, restoring it on exit.

    The original attribute is restored in a ``finally`` so an exception raised
    inside the ``with`` body (e.g. a kernel failure, or a non-reentrant
    activation-checkpoint recompute that ends early) cannot leave the patch
    installed for the rest of the process.
    """
    old = getattr(obj, name)
    setattr(obj, name, new)
    try:
        yield new
    finally:
        setattr(obj, name, old)


def fake_quant_fp4_stacked(
    x: torch.Tensor,
    amax: torch.Tensor,
    block_size: int = FP4_BLOCK_SIZE,
) -> torch.Tensor:
    """FP4 fake quant of a stacked tensor with one global amax per slice of dim 0.

    ``x`` has shape ``(E, ..., N)`` and ``amax`` holds one value per index of
    dim 0 (or a single value shared by all slices). Each slice is quantized
    exactly as a separate :func:`fake_quant_fp4` call with its own amax would
    be, but in one pass. The per-block ue8m0 scale is derived from the slice's
    global amax, matching the per-tensor amax convention of the NVFP4 kernel.
    """
    if x.dim() == 0:
        raise ValueError("fake_quant_fp4_stacked expects a tensor with at least one dim")
    if x.shape[0] == 0:
        # No slices: nothing to quantize, and no kernel launch.
        return x.clone()
    amax = amax.reshape(-1)
    if amax.numel() == 1:
        amax = amax.expand(x.shape[0])
    elif amax.numel() != x.shape[0]:
        raise ValueError(
            f"amax must hold 1 value or one per index of dim 0 ({x.shape[0]}), "
            f"got {amax.numel()}."
        )
    # One global scale per slice, broadcast over the trailing dims.
    global_scale = (amax.float() / FP4_MAX).clamp_min(FP4_AMAX_FLOOR)
    global_scale = ceil_pow2(global_scale)
    view_shape = (x.shape[0],) + (1,) * (x.dim() - 1)
    scaled = x.float() / global_scale.reshape(view_shape)
    scaled = scaled.unflatten(-1, (-1, block_size))
    deq = round_fp4(scaled.clamp(-FP4_MAX, FP4_MAX)) * global_scale.reshape(view_shape).unsqueeze(-1)
    return deq.flatten(-2).to(x.dtype)
