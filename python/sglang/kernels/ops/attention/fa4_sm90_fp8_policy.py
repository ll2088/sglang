# Copyright (c) 2026, SGLang Team.
"""Pure routing policy for the FA4 SM90 d512 FP8-MMA path.

This module must stay importable without torch or sglang so the routing,
descale-normalization and split-selection logic can be unit tested on a
CPU-only runner that only has pytest installed.
"""

from __future__ import annotations

from typing import Optional

SM90_FP8_MMA_HEAD_DIM = 512
SM90_FP8_MMA_MAX_SPLITS = 4
SM90_FP8_MMA_MIN_TILES_PER_SPLIT = 4
SM90_FP8_MMA_TILE_N = 64


def _device_capability_major(device_capability_major: Optional[int]) -> Optional[int]:
    if device_capability_major is not None:
        return int(device_capability_major)
    try:
        import torch  # local import: keep module torch-free at import time
    except ImportError:
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
    """Return True when the call should take the SM90 d512 FP8-MMA kernel."""
    try:
        import torch
    except ImportError:
        return False
    if _device_capability_major(device_capability_major) != 9:
        return False
    if k.dtype != torch.float8_e4m3fn or v.dtype != torch.float8_e4m3fn:
        return False
    if q.dtype not in (torch.float16, torch.bfloat16):
        return False
    if not (q.shape[-1] == k.shape[-1] == v.shape[-1] == SM90_FP8_MMA_HEAD_DIM):
        return False
    if not causal:
        window = window_size or (None, None)
        left = window[0] if len(window) > 0 else None
        right = window[1] if len(window) > 1 else None
        if (left is None or left < 0) and (right is None or right < 0):
            raise NotImplementedError(
                "SM90 FP8 MMA noncausal full attention has known accuracy errors."
            )
    return True


def normalize_kv_descale(descale, *, batch: int, heads: int):
    """Normalize a K/V descale to a contiguous f32 (batch, heads) tensor.

    Kept torch-optional: a scalar descale is returned as a plain float so the
    routing policy stays importable on a CPU-only runner without torch.
    """
    if descale is None:
        return None
    if isinstance(descale, (int, float)):
        return float(descale)
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch is a hard dep in-tree
        raise RuntimeError(
            "normalize_kv_descale needs torch for tensor descales"
        ) from exc

    if not isinstance(descale, torch.Tensor):
        raise ValueError(f"Unsupported descale type: {type(descale)!r}")
    t = descale.to(dtype=torch.float32)
    if t.ndim == 0:
        return t.expand(batch, heads).contiguous()
    if t.ndim == 1:
        if t.numel() == 1:
            return t.expand(batch, heads).contiguous()
        if t.numel() == heads:
            return t.reshape(1, heads).expand(batch, heads).contiguous()
        raise ValueError(f"1-D descale must have 1 or {heads} elements, got {t.numel()}")
    if t.ndim == 2:
        if tuple(t.shape) == (batch, heads):
            return t.contiguous()
        if t.shape[0] == 1 and t.shape[1] == heads:
            return t.expand(batch, heads).contiguous()
        raise ValueError(f"2-D descale must be (1, {heads}) or ({batch}, {heads}), got {tuple(t.shape)}")
    if t.ndim == 4:
        if t.shape[0] == 1 and t.shape[1] == 1 and t.shape[2] == 1 and t.shape[3] == 1:
            return t.reshape(1, 1).expand(batch, heads).contiguous()
        if t.shape[0] == batch and t.shape[1] == heads and t.shape[2] == 1 and t.shape[3] == 1:
            return t.reshape(batch, heads).contiguous()
        raise ValueError(f"4-D descale must be (1,1,1,1) or ({batch},{heads},1,1), got {tuple(t.shape)}")
    raise ValueError(f"Unsupported descale ndim={t.ndim}")


def select_num_splits(
    *,
    fa_version: int,
    max_seqlen_k: int,
    num_heads: int,
    num_kv_heads: int,
    num_sms: int,
    is_fp8_mma: bool,
) -> int:
    """Return the number of KV splits for auto mode (0 means kernel default)."""
    if fa_version == 3:
        return 32
    if not is_fp8_mma:
        return 0
    if max_seqlen_k <= 0 or num_sms <= 0 or num_heads <= 0 or num_kv_heads <= 0:
        return 1
    num_tiles = (max_seqlen_k + SM90_FP8_MMA_TILE_N - 1) // SM90_FP8_MMA_TILE_N
    max_by_tiles = max(1, num_tiles // SM90_FP8_MMA_MIN_TILES_PER_SPLIT)
    qhead_per_kvhead = max(1, num_heads // num_kv_heads)
    max_by_sms = max(1, num_sms // max(1, num_kv_heads * qhead_per_kvhead))
    return max(1, min(SM90_FP8_MMA_MAX_SPLITS, max_by_tiles, max_by_sms))
