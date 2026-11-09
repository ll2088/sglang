# Copyright (c) 2026, SGLang Team.
"""Pure routing policy for the FA4 SM90 d512 FP8-MMA path.

This module must stay importable without torch, CUDA, or the sglang package
so routing/descale/split logic can be unit-tested on a CPU-only runner.
"""

from __future__ import annotations

from typing import Optional

SM90_FP8_MMA_HEAD_DIM = 512
SM90_FP8_MMA_MAJOR = 9
SM90_FP8_MMA_MAX_SPLITS = 4
FA3_MAX_SPLITS = 32
SPLIT_TILE_N = 256


def _is_fp8_e4m3(dtype) -> bool:
    return getattr(dtype, "name", "") == "float8_e4m3fn"


def is_sm90_fp8_kv_mma(
    q,
    k,
    v,
    causal: bool,
    window_size,
    *,
    device_capability_major: Optional[int] = None,
) -> bool:
    """Return whether the call qualifies for the SM90 d512 FP8-MMA kernel."""
    if device_capability_major is None:
        import torch

        device_capability_major = torch.cuda.get_device_capability()[0]
    if device_capability_major != SM90_FP8_MMA_MAJOR:
        return False
    if not (_is_fp8_e4m3(k.dtype) and _is_fp8_e4m3(v.dtype)):
        return False
    if not (q.shape[-1] == k.shape[-1] == v.shape[-1] == SM90_FP8_MMA_HEAD_DIM):
        return False
    if not causal:
        left, right = window_size if window_size is not None else (None, None)
        left = -1 if left is None else left
        right = -1 if right is None else right
        if left < 0 and right < 0:
            return False
    return True


def normalize_kv_descale(descale, *, batch: int, heads: int):
    """Normalize a K/V descale to a contiguous f32 (batch, heads) tensor."""
    if descale is None:
        return None
    import torch

    if isinstance(descale, (int, float)):
        return torch.full((batch, heads), float(descale), dtype=torch.float32)
    if not isinstance(descale, torch.Tensor):
        raise ValueError(f"unsupported descale type: {type(descale)!r}")
    if descale.ndim == 1:
        if descale.numel() != heads:
            raise ValueError(
                f"1-D descale must have {heads} elements, got {descale.numel()}"
            )
        out = descale.reshape(1, heads).expand(batch, heads)
    elif descale.ndim == 2:
        if tuple(descale.shape) != (batch, heads):
            raise ValueError(
                f"2-D descale must be {(batch, heads)}, got {tuple(descale.shape)}"
            )
        out = descale
    elif descale.ndim == 4 and tuple(descale.shape[-2:]) == (1, 1):
        if tuple(descale.shape[:2]) != (batch, heads):
            raise ValueError(
                f"4-D descale must be {(batch, heads, 1, 1)}, "
                f"got {tuple(descale.shape)}"
            )
        out = descale.reshape(batch, heads)
    else:
        raise ValueError(f"unsupported descale shape: {tuple(descale.shape)}")
    return out.to(dtype=torch.float32).contiguous()


def select_num_splits(
    *,
    fa_version: int,
    max_seqlen_k: int,
    num_heads: int,
    num_kv_heads: int,
    num_sms: int,
    is_fp8_mma: bool,
) -> int:
    """Pick a SplitKV count; FA3 keeps its cap, FA4 FP8-MMA auto-selects."""
    if fa_version == 3:
        return FA3_MAX_SPLITS
    if not is_fp8_mma:
        return 0
    if max_seqlen_k <= 0 or num_sms <= 0 or num_heads <= 0:
        return 1
    num_tiles = (max_seqlen_k + SPLIT_TILE_N - 1) // SPLIT_TILE_N
    if num_tiles <= 1:
        return 1
    qhead_ratio = max(1, num_heads // max(1, num_kv_heads))
    wanted = max(1, num_sms // max(1, qhead_ratio))
    return max(1, min(SM90_FP8_MMA_MAX_SPLITS, num_tiles, wanted))
