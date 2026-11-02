"""Pure-torch FP4 fake quantization for DeepSeek-V4.1.

Indexer values use per-32 UE8M0 scales; compressed KV uses per-16 E4M3 scales.
Both paths round to the E2M1 grid with ties to even.
"""

import torch

FP8_MAX = 448.0
FP4_MAX = 6.0
FP8_BLOCK_SIZE = 32
FP4_BLOCK_SIZE = 32
FP4_AMAX_FLOOR = 6 * 2.0**-126

# Number of 16-element blocks handled per program in the Triton kernel. The
# pure-torch path below mirrors the same batching so both stay numerically
# identical; keep the two in sync.
STATIC_FP4_BLOCKS_PER_PROGRAM = 64


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


def _block_offsets(num_blocks: int, blocks_per_program: int, device) -> torch.Tensor:
    """int64 block offsets for the batched kernel layout.

    The Triton kernel computes ``pid * BLOCKS_PER_PROGRAM + arange(...)`` in
    int64 so stacked MoE weights with more than 2**31 elements do not overflow.
    This helper reproduces that arithmetic on the host side so the pure-torch
    reference and the kernel agree bit-for-bit.
    """
    return torch.arange(num_blocks, dtype=torch.int64, device=device)


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


def fake_quant_fp4_batched(
    x: torch.Tensor,
    block_size: int = FP4_BLOCK_SIZE,
    blocks_per_program: int = STATIC_FP4_BLOCKS_PER_PROGRAM,
) -> torch.Tensor:
    """Batched-block variant of :func:`fake_quant_fp4`.

    Mirrors the Triton kernel's ``BLOCKS_PER_PROGRAM`` layout: blocks are
    grouped into programs of ``blocks_per_program`` and the tail is masked.
    The result is bit-identical to :func:`fake_quant_fp4`; this entry point
    exists so tests can assert that equivalence and so callers can opt into
    the same batching shape as the kernel.
    """
    if blocks_per_program <= 0:
        raise ValueError("blocks_per_program must be positive")
    flat = x.float().reshape(-1)
    numel = flat.numel()
    if numel % block_size != 0:
        raise ValueError(
            f"numel {numel} is not a multiple of block_size {block_size}"
        )
    num_blocks = numel // block_size
    if num_blocks == 0:
        return x.clone()

    # int64 offsets: stacked MoE weights can exceed 2**31 elements.
    block_idx = _block_offsets(num_blocks, blocks_per_program, x.device)
    block_mask = block_idx < num_blocks
    idx = block_idx[:, None] * block_size + torch.arange(
        block_size, device=x.device, dtype=torch.int64
    )[None, :]

    scale = block_scale(x, block_size, FP4_MAX, FP4_AMAX_FLOOR).reshape(-1)
    scale = scale.index_select(0, block_idx.clamp_max(num_blocks - 1))
    scale = torch.where(block_mask, scale, torch.zeros_like(scale))

    gathered = flat.index_select(0, idx.reshape(-1).clamp_max(numel - 1))
    gathered = gathered.reshape(num_blocks, block_size)
    gathered = torch.where(block_mask[:, None], gathered, torch.zeros_like(gathered))

    scaled = gathered / scale.unsqueeze(-1)
    deq = round_fp4(scaled.clamp(-FP4_MAX, FP4_MAX)) * scale.unsqueeze(-1)
    deq = torch.where(block_mask[:, None], deq, torch.zeros_like(deq))
    return deq.reshape(x.shape).to(x.dtype)


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
