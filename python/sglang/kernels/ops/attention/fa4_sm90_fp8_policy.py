# Copyright (c) 2026, SGLang Team.
"""Pure routing policy for the FA4 SM90 d512 FP8-MMA path.

This module must stay importable without torch or sglang so the routing
checks can run on a CPU-only, minimal-dependency runner.
"""

from __future__ import annotations

from typing import Optional

SM90_FP8_MMA_HEAD_DIM = 512
SM90_FP8_MMA_MAX_SPLITS = 4
FA3_MAX_SPLITS = 32
SPLIT_TILE_N = 64


def _device_capability_major(device_capability_major: Optional[int]) -> Optional[int]:
    if device_capability_major is not None:
        return int(device_capability_major)
    try:
        import torch  # local import: keep module import torch-free
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return int(torch.cuda.get_device_capability()[0])


def is_sm90_fp8_kv_mma(
    q,
    k,
    v,
    causal: bool,
    window_size,
    *,
    device_capability_major: Optional[int] = None,
) -> bool:
    """Return whether the call should take the SM90 d512 FP8-MMA kernel."""
    major = _device_capability_major(device_capability_major)
    if major != 9:
        return False
    try:
        import torch
    except ImportError:
        return False
    if k.dtype != torch.float8_e4m3fn or v.dtype != torch.float8_e4m3fn:
        return False
    if q.dtype not in (torch.float16, torch.bfloat16):
        return False
    if not (q.shape[-1] == k.shape[-1] == v.shape[-1] == SM90_FP8_MMA_HEAD_DIM):
        return False
    if not causal:
        left, right = (window_size or (None, None))[:2]
        if left is None and right is None:
            raise NotImplementedError(
                "SM90 FP8 MMA noncausal full attention has known accuracy errors."
            )
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
    t = descale.to(dtype=torch.float32)
    if t.ndim == 0:
        return t.expand(batch, heads).contiguous()
    if t.ndim == 1:
        if t.numel() != heads:
            raise ValueError(f"descale 1-D length {t.numel()} != heads {heads}")
        return t.reshape(1, heads).expand(batch, heads).contiguous()
    if t.ndim == 2:
        if tuple(t.shape) != (batch, heads):
            raise ValueError(f"descale shape {tuple(t.shape)} != {(batch, heads)}")
        return t.contiguous()
    if t.ndim == 4 and tuple(t.shape) == (batch, heads, 1, 1):
        return t.reshape(batch, heads).contiguous()
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
    """Pick a SplitKV count; 0 means the kernel heuristic."""
    if fa_version == 3:
        return FA3_MAX_SPLITS
    if fa_version != 4:
        return 0
    if not is_fp8_mma:
        return 0
    if max_seqlen_k <= 0 or num_sms <= 0 or num_heads <= 0 or num_kv_heads <= 0:
        return 1
    num_tiles = (max_seqlen_k + SPLIT_TILE_N - 1) // SPLIT_TILE_N
    if num_tiles <= 1:
        return 1
    qhead_per_kvhead = max(1, num_heads // num_kv_heads)
    target = max(1, num_sms // max(1, qhead_per_kvhead))
    return max(1, min(SM90_FP8_MMA_MAX_SPLITS, num_tiles, target))
