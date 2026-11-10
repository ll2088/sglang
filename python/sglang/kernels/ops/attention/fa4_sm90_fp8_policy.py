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


def _dtype_name(dtype) -> str:
    """Return a stable dtype name without importing torch."""
    if dtype is None:
        return ""
    name = getattr(dtype, "__name__", None)
    if name:
        return str(name)
    return str(dtype).rsplit(".", 1)[-1]


def _is_fp8_e4m3(dtype) -> bool:
    return _dtype_name(dtype) in ("float8_e4m3fn", "float8_e4m3fnuz")


def _is_fp16_or_bf16(dtype) -> bool:
    return _dtype_name(dtype) in ("float16", "bfloat16", "half", "BFloat16", "Half")


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
    if _device_capability_major(device_capability_major) != 9:
        return False
    if not _is_fp8_e4m3(k.dtype) or not _is_fp8_e4m3(v.dtype):
        return False
    if not _is_fp16_or_bf16(q.dtype):
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


def is_sm90_fp8_kv_mma_dtypes(
    q_dtype,
    k_dtype,
    v_dtype,
    q_head_dim: int,
    k_head_dim: int,
    v_head_dim: int,
    causal: bool,
    window_size,
    *,
    device_capability_major: Optional[int] = None,
) -> bool:
    """Torch-free variant of :func:`is_sm90_fp8_kv_mma` for host-side tests."""
    if _device_capability_major(device_capability_major) != 9:
        return False
    if not _is_fp8_e4m3(k_dtype) or not _is_fp8_e4m3(v_dtype):
        return False
    if not _is_fp16_or_bf16(q_dtype):
        return False
    if not (q_head_dim == k_head_dim == v_head_dim == SM90_FP8_MMA_HEAD_DIM):
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
    except ImportError:
        # Torch-free fallback: accept nested Python sequences so the routing
        # policy stays testable on a CPU-only runner without torch.
        return _normalize_kv_descale_py(descale, batch=batch, heads=heads)

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


def _normalize_kv_descale_py(descale, *, batch: int, heads: int):
    """Torch-free descale normalization for nested Python sequences."""
    if isinstance(descale, (list, tuple)):
        if len(descale) == 1 and isinstance(descale[0], (list, tuple)):
            row = list(descale[0])
            if len(row) == 1:
                return [[float(row[0])] * heads for _ in range(batch)]
            if len(row) == heads:
                return [[float(x) for x in row] for _ in range(batch)]
            raise ValueError(
                f"2-D descale must be (1, {heads}) or ({batch}, {heads}), got (1, {len(row)})"
            )
        if len(descale) == batch and all(
            isinstance(r, (list, tuple)) and len(r) == heads for r in descale
        ):
            return [[float(x) for x in r] for r in descale]
        if len(descale) == 1:
            return [[float(descale[0])] * heads for _ in range(batch)]
        if len(descale) == heads:
            row = [float(x) for x in descale]
            return [list(row) for _ in range(batch)]
        raise ValueError(
            f"1-D descale must have 1 or {heads} elements, got {len(descale)}"
        )
    raise ValueError(f"Unsupported descale type: {type(descale)!r}")


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
