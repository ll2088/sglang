"""Convert MXFP8 weights to block-fp8 [128,128] for AMD gfx942 (CDNA3 / MI300).

gfx942 has no hardware MX-scaled matmul: Triton's ``tl.dot_scaled`` fails to
lower and the gfx950 ``mfma_scale`` intrinsics are unavailable. So MXFP8
checkpoints (e4m3fn weights + 1x32 UE8M0 scales) are converted at load time to
block-wise FP8 [128,128] (e4m3fn + fp32 scales), which runs through SGLang's
native DeepSeek-V3 block-fp8 kernels (aiter / triton). The conversion is:

    bf16 = e4m3.to(f32) * exp2(ue8m0_scale.to(f32) - 127.0)   # dequant 1x32
    block-fp8 = per-128x128-block quantize(bf16)              # requant 128x128
"""

from __future__ import annotations

from typing import Tuple

import torch

MXFP8_BLOCK_SIZE = 32


def _ue8m0_to_fp32(scale_u8: torch.Tensor) -> torch.Tensor:
    """UE8M0 uint8 (biased exponent, bias 127) -> fp32 multiplier 2^(v-127)."""
    return (scale_u8.to(torch.int32) << 23).view(torch.float32)


def dequant_mxfp8_2d_to_bf16(
    weight: torch.Tensor, scale_u8: torch.Tensor
) -> torch.Tensor:
    """Dequant a 2D MXFP8 tensor (e4m3fn + 1x32 UE8M0 scales) to bf16.

    weight: [N, K] float8_e4m3fn; scale_u8: [N, K//32] uint8.
    """
    n, k = weight.shape
    descale = _ue8m0_to_fp32(scale_u8).unsqueeze(-1)  # [N, K//32, 1]
    deq = weight.to(torch.float32).view(n, k // MXFP8_BLOCK_SIZE, MXFP8_BLOCK_SIZE)
    return (deq * descale).view(n, k).to(torch.bfloat16)


def bf16_to_block_fp8_128(
    weight: torch.Tensor, block: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D bf16/fp32 weight to block-wise FP8 (e4m3fn) + fp32 scales.

    Returns (qweight [N,K] e4m3fn, scale [ceil(N/block), ceil(K/block)] fp32).
    Mirrors the DeepSeek-V3 block-fp8 contract (divide by e4m3fn max 448).
    The downstream gfx942 path normalizes e4m3fn -> e4m3fnuz separately.
    """
    n, k = weight.shape
    pn = ((n + block - 1) // block) * block
    pk = ((k + block - 1) // block) * block
    xp = torch.zeros((pn, pk), dtype=torch.float32, device=weight.device)
    xp[:n, :k] = weight.to(torch.float32)
    xv = xp.view(pn // block, block, pk // block, block)
    amax = xv.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-4)
    sf = amax / 448.0
    xq = (xv / sf).to(torch.float8_e4m3fn)
    qweight = xq.view(pn, pk)[:n, :k].contiguous()
    scale = sf.view(pn // block, pk // block).contiguous()
    return qweight, scale


def requant_block_fp8_to_mxfp8(
    weight: torch.Tensor, scale: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Requantize block-FP8 weights to E4M3 with one UE8M0 scale per 1x32.

    Inverse direction of :func:`convert_mxfp8_weight_to_block_fp8`: a
    checkpoint stored as E4M3 with one float32 scale per ``[block_m, block_k]``
    block is dequantized and re-quantized to the MXFP8 layout consumed by
    block-scaled MMA kernels (power-of-two scale per row and 32 columns).

    Args:
        weight: ``[N, K]`` E4M3 values.
        scale: ``[ceil(N/block_m), ceil(K/block_k)]`` float32 dequantization
            scales (``real = weight * scale``).

    Returns:
        ``(qweight, scale_u8)``: E4M3 weights ``[N, K]`` and UE8M0 scale bytes
        ``[N, K // 32]`` uint8.
    """
    n, k = weight.shape
    if k % MXFP8_BLOCK_SIZE != 0:
        raise ValueError(
            f"requant_block_fp8_to_mxfp8 requires K divisible by "
            f"{MXFP8_BLOCK_SIZE}, got {k}."
        )
    blocks_m, blocks_k = scale.shape
    block_m = (n + blocks_m - 1) // blocks_m
    block_k = (k + blocks_k - 1) // blocks_k
    if block_m * blocks_m < n or block_k * blocks_k < k:
        raise ValueError(
            f"requant_block_fp8_to_mxfp8 scale shape {tuple(scale.shape)} "
            f"does not cover weight shape {(n, k)}."
        )
    real = weight.to(torch.float32).view(
        blocks_m, block_m, blocks_k, block_k
    )
    real = real * scale[:, None, :, None]
    real = real.reshape(blocks_m * block_m, blocks_k * block_k)[:n, :k]
    return _bf16_to_mxfp8_1x32(real)


def _bf16_to_mxfp8_1x32(
    weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D fp32/bf16 tensor to E4M3 + UE8M0 scales per 1x32 block."""
    n, k = weight.shape
    x = weight.to(torch.float32).view(n, k // MXFP8_BLOCK_SIZE, MXFP8_BLOCK_SIZE)
    # Non-finite inputs must not poison the scale: treat them as zero so the
    # output stays finite and the caller can detect the bad block via amax.
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    amax = x.abs().amax(dim=-1).clamp(min=1e-30)
    # Round the scale up to the next power of two so the mantissa fits E4M3.
    scale_biased = torch.ceil(torch.log2(amax / 448.0)) + 127.0
    scale_biased = scale_biased.clamp(min=0.0, max=254.0)
    descale = torch.exp2(scale_biased - 127.0)
    xq = (x / descale.unsqueeze(-1)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return xq.view(n, k).contiguous(), scale_biased.to(torch.uint8)


def convert_mxfp8_weight_to_block_fp8(
    weight: torch.Tensor, scale_u8: torch.Tensor, block: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    """MXFP8 (e4m3fn + 1x32 UE8M0) -> block-fp8 [block,block] (e4m3fn + fp32).

    Used on gfx942 to run MXFP8 checkpoints through the fast native block-fp8
    kernels.
    """
    bf16 = dequant_mxfp8_2d_to_bf16(weight, scale_u8)
    return bf16_to_block_fp8_128(bf16, block=block)
