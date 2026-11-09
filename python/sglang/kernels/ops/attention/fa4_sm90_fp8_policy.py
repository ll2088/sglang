# Copyright (c) 2026, SGLang Team.
"""Pure routing policy for the FA4 SM90 d512 FP8-MMA path.

This module must stay importable with the Python standard library only: it is
loaded by path from CPU-only CI runners before sglang (and torch) are
installed. Keep all torch access lazy and inside helpers.
"""

from __future__ import annotations

from typing import Optional

SM90_FP8_MMA_HEAD_DIM = 512
SM90_FP8_MMA_MAJOR = 9
SPLIT_TILE_N = 128
SPLIT_MAX = 32
SPLIT_FP8_MMA_MAX = 4


def _device_major(device_capability_major: Optional[int]) -> Optional[int]:
    if device_capability_major is not None:
        return int(device_capability_major)
    try:
        import torch  # local import: keep module torch-free at import time
    except Exception:
        return None
    try:
        return int(torch.cuda.get_device_capability()[0])
    except Exception:
        return None


def is_sm90_fp8_kv_mma(
    q,
    k,
    v,
    causal: bool,
    window_size,
    *,
    device_capability_major: Optional[int] = None,
) -> bool:
    """Return True when the call must take the SM90 d512 FP8-MMA kernel.

    Requirements: SM90, E4M3 K/V, equal d512 head dims, and either causal or a
    bounded (sliding-window) mask. Noncausal full attention is rejected because
    the kernel has known accuracy errors there.
    """
    try:
        import torch
    except Exception:
        return False
    if _device_major(device_capability_major) != SM90_FP8_MMA_MAJOR:
        return False
    if k.dtype != torch.float8_e4m3fn or v.dtype != torch.float8_e4m3fn:
        return False
    if q.shape[-1] != SM90_FP8_MMA_HEAD_DIM:
        return False
    if k.shape[-1] != SM90_FP8_MMA_HEAD_DIM or v.shape[-1] != SM90_FP8_MMA_HEAD_DIM:
        return False
    if causal:
        return True
    left, right = (window_size or (None, None))[:2]
    return left is not None or right is not None


def normalize_kv_descale(descale, *, batch: int, heads: int):
    """Normalize a K/V descale to a contiguous f32 (batch, heads) tensor."""
    if descale is None:
        return None
    import torch

    if isinstance(descale, (int, float)):
        return torch.full((batch, heads), float(descale), dtype=torch.float32)
    if not isinstance(descale, torch.Tensor):
        raise ValueError(f"unsupported descale type: {type(descale)!r}")
    t = descale
    if t.ndim == 0:
        return t.to(torch.float32).expand(batch, heads).contiguous()
    if t.ndim == 1:
        if t.numel() != heads:
            raise ValueError(f"descale 1-D length {t.numel()} != heads {heads}")
        return t.to(torch.float32).reshape(1, heads).expand(batch, heads).contiguous()
    if t.ndim == 2:
        if tuple(t.shape) != (batch, heads):
            raise ValueError(
                f"descale shape {tuple(t.shape)} != ({batch}, {heads})"
            )
        return t.to(torch.float32).contiguous()
    if t.ndim == 4 and t.shape[-1] == 1 and t.shape[-2] == 1:
        if tuple(t.shape[:2]) != (batch, heads):
            raise ValueError(
                f"descale shape {tuple(t.shape)} != ({batch}, {heads}, 1, 1)"
            )
        return t.to(torch.float32).reshape(batch, heads).contiguous()
    raise ValueError(f"unsupported descale shape {tuple(t.shape)}")


def select_num_splits(
    *,
    fa_version: int,
    max_seqlen_k: int,
    num_heads: int,
    num_kv_heads: int,
    num_sms: int,
    is_fp8_mma: bool,
) -> int:
    """Pick a SplitKV count; 0 means "let the kernel auto-select"."""
    if fa_version == 3:
        return SPLIT_MAX
    if fa_version != 4:
        return 1
    if max_seqlen_k <= 0 or num_sms <= 0 or num_heads <= 0 or num_kv_heads <= 0:
        return 1
    if not is_fp8_mma:
        return 0
    num_tiles = (max_seqlen_k + SPLIT_TILE_N - 1) // SPLIT_TILE_N
    if num_tiles <= 1:
        return 1
    return max(1, min(SPLIT_FP8_MMA_MAX, num_tiles))
