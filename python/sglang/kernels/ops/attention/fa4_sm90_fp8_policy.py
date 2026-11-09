# Copyright (c) 2026, SGLang Team.
"""Pure routing policy for the FA4 SM90 d512 FP8-MMA path.

This module must stay importable without torch, sglang, or CUDA so the
routing / descale / split logic can be unit-tested on a CPU-only runner.
"""

from typing import Optional

SM90_FP8_MMA_HEAD_DIM = 512
SM90_FP8_MMA_MAJOR = 9

# SplitKV auto-selection bounds.
FA3_MAX_SPLITS = 32
FA4_FP8_MMA_MAX_SPLITS = 4
FA4_MIN_SPLITS = 1
SPLIT_TILE_N = 64
SPLIT_MIN_SEQLEN_K = 128


def _is_e4m3(dtype) -> bool:
    return getattr(dtype, "__name__", "") == "float8_e4m3fn"


def _head_dim(t) -> Optional[int]:
    shape = getattr(t, "shape", None)
    if shape is None or len(shape) == 0:
        return None
    return int(shape[-1])


def _device_major(device_capability_major: Optional[int]) -> Optional[int]:
    if device_capability_major is not None:
        return int(device_capability_major)
    try:
        import torch  # noqa: PLC0415 - optional, keeps module torch-free
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
    device_capability_major: Optional[int] = None,
) -> bool:
    """Return whether the call qualifies for the SM90 d512 FP8-MMA kernel.

    Requires SM90, E4M3 K/V, equal q/k/v head dim of 512, and either causal
    or a sliding window. Noncausal full attention is rejected because the
    SM90 FP8 MMA kernel has known accuracy errors there.
    """
    if _device_major(device_capability_major) != SM90_FP8_MMA_MAJOR:
        return False
    if not (_is_e4m3(getattr(k, "dtype", None)) and _is_e4m3(getattr(v, "dtype", None))):
        return False
    dims = (_head_dim(q), _head_dim(k), _head_dim(v))
    if dims != (SM90_FP8_MMA_HEAD_DIM,) * 3:
        return False
    if causal:
        return True
    left, right = (window_size if window_size is not None else (None, None))
    return left is not None or right is not None


def normalize_kv_descale(descale, *, batch: int, heads: int):
    """Normalize a K/V descale to a contiguous f32 (batch, heads) tensor.

    Accepts a Python scalar, a 1-D per-head vector, a 2-D (batch, heads)
    tensor, or a 4-D (batch, heads, 1, 1) tensor. ``None`` passes through.
    """
    if descale is None:
        return None
    import torch  # noqa: PLC0415 - only needed when a descale is present

    if isinstance(descale, (int, float)):
        return torch.full((batch, heads), float(descale), dtype=torch.float32)
    if not isinstance(descale, torch.Tensor):
        raise ValueError(f"unsupported descale type: {type(descale)!r}")
    out = descale.to(dtype=torch.float32)
    if out.ndim == 1:
        if out.shape[0] != heads:
            raise ValueError(
                f"1-D descale must have {heads} entries, got {tuple(out.shape)}"
            )
        out = out.reshape(1, heads).expand(batch, heads)
    elif out.ndim == 2:
        if tuple(out.shape) != (batch, heads):
            raise ValueError(
                f"2-D descale must be {(batch, heads)}, got {tuple(out.shape)}"
            )
    elif out.ndim == 4:
        if tuple(out.shape) != (batch, heads, 1, 1):
            raise ValueError(
                f"4-D descale must be {(batch, heads, 1, 1)}, got {tuple(out.shape)}"
            )
        out = out.reshape(batch, heads)
    else:
        raise ValueError(f"unsupported descale ndim: {out.ndim}")
    return out.contiguous()


def select_num_splits(
    *,
    fa_version: int,
    max_seqlen_k: int,
    num_heads: int,
    num_kv_heads: int,
    num_sms: int,
    is_fp8_mma: bool,
) -> int:
    """Pick the SplitKV count for auto mode (``num_splits == 0``).

    FA3 keeps its historical cap of 32. FA4 full attention is uncapped (0),
    while the SM90 FP8-MMA path uses a small bounded split count.
    """
    if fa_version == 3:
        return FA3_MAX_SPLITS
    if fa_version != 4:
        return FA4_MIN_SPLITS
    if not is_fp8_mma:
        return 0
    if max_seqlen_k <= 0 or num_sms <= 0 or num_heads <= 0:
        return FA4_MIN_SPLITS
    if max_seqlen_k < SPLIT_MIN_SEQLEN_K:
        return FA4_MIN_SPLITS
    num_tiles = (max_seqlen_k + SPLIT_TILE_N - 1) // SPLIT_TILE_N
    qhead_ratio = max(1, num_heads // max(1, num_kv_heads))
    splits = min(FA4_FP8_MMA_MAX_SPLITS, num_tiles, max(1, num_sms // qhead_ratio))
    return max(FA4_MIN_SPLITS, splits)
