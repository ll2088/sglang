# Copyright (c) 2026, SGLang Team.
"""Pure routing policy for the FA4 SM90 d512 FP8-MMA path.

This module must stay free of torch and sglang imports at module scope so the
routing checks can run on a CPU-only runner without installing sglang.
"""

from __future__ import annotations

from typing import Optional

SM90_FP8_MMA_HEAD_DIM = 512
SM90_FP8_MMA_ARCH_MAJOR = 9
SM90_FP8_MMA_BLOCK_ALIGN = 64
SM90_FP8_MMA_MAX_SPLITS = 4
SM90_FP8_MMA_MIN_SPLIT_SEQLEN = 128
SM90_FP8_MMA_TILE_N = 128
FA3_MAX_SPLITS = 32


def _is_e4m3(dtype) -> bool:
    return getattr(dtype, "__name__", "") == "float8_e4m3fn"


def _head_dim(t) -> Optional[int]:
    shape = getattr(t, "shape", None)
    return None if shape is None else int(shape[-1])


def _arch_major(device_capability_major: Optional[int]) -> Optional[int]:
    if device_capability_major is not None:
        return int(device_capability_major)
    try:
        import torch  # noqa: PLC0415 - lazy so module import stays torch-free
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
    """Return whether the call qualifies for the SM90 d512 FP8-MMA kernel."""
    if _arch_major(device_capability_major) != SM90_FP8_MMA_ARCH_MAJOR:
        return False
    if not _is_e4m3(getattr(k, "dtype", None)):
        return False
    if not _is_e4m3(getattr(v, "dtype", None)):
        return False
    qd, kd, vd = _head_dim(q), _head_dim(k), _head_dim(v)
    if qd != SM90_FP8_MMA_HEAD_DIM or kd != qd or vd != qd:
        return False
    if not causal:
        left, right = (None, None) if window_size is None else window_size
        if (left is None or left < 0) and (right is None or right < 0):
            # Noncausal full attention has known accuracy errors on this path.
            return False
    return True


def normalize_kv_descale(descale, *, batch: int, heads: int):
    """Normalize a K/V descale to a contiguous f32 (batch, heads) tensor."""
    if descale is None:
        return None
    import torch  # noqa: PLC0415 - lazy so module import stays torch-free

    target = (int(batch), int(heads))
    if isinstance(descale, (int, float)):
        return torch.full(target, float(descale), dtype=torch.float32)
    if not isinstance(descale, torch.Tensor):
        raise ValueError(f"unsupported descale type: {type(descale)!r}")
    t = descale.to(dtype=torch.float32)
    if t.ndim == 1:
        if t.shape[0] != target[1]:
            raise ValueError(f"descale shape {tuple(t.shape)} != {target}")
        t = t.reshape(1, target[1]).expand(target)
    elif t.ndim == 2:
        if t.shape != target:
            raise ValueError(f"descale shape {tuple(t.shape)} != {target}")
    elif t.ndim == 4:
        if t.shape[2:] != (1, 1) or t.shape[:2] != target:
            raise ValueError(f"descale shape {tuple(t.shape)} != {target}")
        t = t.reshape(target)
    else:
        raise ValueError(f"unsupported descale ndim {t.ndim}")
    return t.contiguous()


def select_num_splits(
    *,
    fa_version: int,
    max_seqlen_k: int,
    num_heads: int,
    num_kv_heads: int,
    num_sms: int,
    is_fp8_mma: bool,
) -> int:
    """Pick the SplitKV count; 0 means the kernel heuristic."""
    if fa_version == 3:
        return FA3_MAX_SPLITS
    if fa_version != 4:
        return 0
    if not is_fp8_mma:
        return 0
    if max_seqlen_k <= 0 or num_sms <= 0 or num_heads <= 0:
        return 1
    if max_seqlen_k < SM90_FP8_MMA_MIN_SPLIT_SEQLEN:
        return 1
    num_tiles = (int(max_seqlen_k) + SM90_FP8_MMA_TILE_N - 1) // SM90_FP8_MMA_TILE_N
    qhead_ratio = max(1, int(num_heads) // max(1, int(num_kv_heads)))
    mblocks = max(1, int(num_sms) // qhead_ratio)
    splits = min(SM90_FP8_MMA_MAX_SPLITS, num_tiles, mblocks)
    return max(1, splits)
