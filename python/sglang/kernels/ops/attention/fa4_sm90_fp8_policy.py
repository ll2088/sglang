# Copyright (c) 2026, SGLang Team.
"""Pure routing policy for the FA4 SM90 d512 FP8-MMA path.

This module must stay free of torch and sglang imports at module scope so the
routing rules can be unit-tested on a CPU-only runner without installing the
full sglang dependency set.
"""

from typing import Optional, Tuple

SM90_FP8_MMA_HEAD_DIM = 512
SM90_FP8_MMA_MAJOR = 9
FA3_SPLIT_CAP = 32
FA4_FP8_MMA_MAX_SPLITS = 4
FA4_SPLIT_TILE_N = 256


def _dtype_name(dtype) -> str:
    return getattr(dtype, "__name__", str(dtype))


def _is_e4m3(dtype) -> bool:
    return _dtype_name(dtype) in ("float8_e4m3fn", "float8_e4m3fnuz")


def _is_fp8(dtype) -> bool:
    return _dtype_name(dtype) in (
        "float8_e4m3fn",
        "float8_e4m3fnuz",
        "float8_e5m2",
        "float8_e5m2fnuz",
    )


def _head_dim(t) -> Optional[int]:
    shape = getattr(t, "shape", None)
    if shape is None or len(shape) == 0:
        return None
    return int(shape[-1])


def _device_major(device_capability_major: Optional[int]) -> Optional[int]:
    if device_capability_major is not None:
        return int(device_capability_major)
    try:
        import torch  # local import: keep module import torch-free
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
    window_size: Tuple[Optional[int], Optional[int]],
    *,
    device_capability_major: Optional[int] = None,
) -> bool:
    """Return whether the SM90 d512 FP8-MMA kernel may handle this call.

    Raises NotImplementedError for the noncausal full-attention combination
    that the kernel rejects at dispatch time.
    """
    major = _device_major(device_capability_major)
    if major != SM90_FP8_MMA_MAJOR:
        return False
    if not _is_e4m3(getattr(k, "dtype", None)):
        return False
    if not _is_e4m3(getattr(v, "dtype", None)):
        return False
    if _is_fp8(getattr(q, "dtype", None)):
        return False
    dims = (_head_dim(q), _head_dim(k), _head_dim(v))
    if dims != (SM90_FP8_MMA_HEAD_DIM,) * 3:
        return False
    left, right = window_size if window_size is not None else (None, None)
    is_local = (left is not None and left >= 0) or (right is not None and right >= 0)
    if not causal and not is_local:
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
    t = descale
    if t.ndim == 0:
        t = t.reshape(1, 1).expand(batch, heads)
    elif t.ndim == 1:
        if t.numel() != heads:
            raise ValueError(
                f"1-D descale must have {heads} elements, got {t.numel()}"
            )
        t = t.reshape(1, heads).expand(batch, heads)
    elif t.ndim == 2:
        if tuple(t.shape) != (batch, heads):
            raise ValueError(
                f"2-D descale must be ({batch}, {heads}), got {tuple(t.shape)}"
            )
    elif t.ndim == 4:
        if tuple(t.shape) != (batch, heads, 1, 1):
            raise ValueError(
                f"4-D descale must be ({batch}, {heads}, 1, 1), got {tuple(t.shape)}"
            )
        t = t.reshape(batch, heads)
    else:
        raise ValueError(f"unsupported descale ndim: {t.ndim}")
    return t.to(dtype=torch.float32).contiguous()


def select_num_splits(
    *,
    fa_version: int,
    max_seqlen_k: int,
    num_heads: int,
    num_kv_heads: int,
    num_sms: int,
    is_fp8_mma: bool,
) -> int:
    """Pick the SplitKV count for auto mode (0 means kernel heuristic)."""
    if fa_version == 3:
        return FA3_SPLIT_CAP
    if fa_version != 4:
        return 0
    if not is_fp8_mma:
        return 0
    if max_seqlen_k <= 0 or num_sms <= 0 or num_heads <= 0 or num_kv_heads <= 0:
        return 1
    num_tiles = (max_seqlen_k + FA4_SPLIT_TILE_N - 1) // FA4_SPLIT_TILE_N
    if num_tiles <= 1:
        return 1
    qhead_per_kvhead = max(1, num_heads // num_kv_heads)
    mblocks = num_kv_heads * qhead_per_kvhead
    wanted = max(1, num_sms // max(1, mblocks))
    return max(1, min(FA4_FP8_MMA_MAX_SPLITS, wanted, num_tiles))
