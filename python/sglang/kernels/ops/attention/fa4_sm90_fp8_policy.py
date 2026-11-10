# Copyright (c) 2026, SGLang Team.
"""Pure routing policy for the FA4 SM90 d512 FP8-MMA path.

Torch-free at import time so routing can be unit-tested on CPU without a
compiled FA4 extension. Tensor inputs are duck-typed (``.dtype``/``.shape``).
"""

from typing import Optional, Sequence

FP8_E4M3_DTYPE_NAMES = frozenset(
    {"float8_e4m3fn", "float8_e4m3fnuz", "fp8_e4m3", "fp8_e4m3fn"}
)
SM90_FP8_MMA_HEAD_SIZE = 512
FA3_SPLIT_CAP = 32
FP8_MMA_MAX_SPLITS = 4
FP8_MMA_MIN_SEQLEN_PER_SPLIT = 256


def _dtype_name(dtype) -> str:
    return getattr(dtype, "__name__", str(dtype)).replace("torch.", "")


def _head_size(tensor) -> int:
    shape = getattr(tensor, "shape", None)
    return int(shape[-1]) if shape is not None else 0


def _is_full_window(window_size) -> bool:
    if window_size is None:
        return True
    left, right = window_size
    return left is None and right is None


def is_sm90_fp8_kv_mma_dtypes(
    q_dtype,
    k_dtype,
    v_dtype,
    q_head_size: int,
    k_head_size: int,
    v_head_size: int,
    causal: bool,
    window_size,
    *,
    device_capability_major: Optional[int],
) -> bool:
    """Return whether the SM90 d512 FP8-MMA kernel handles this combination."""
    if device_capability_major != 9:
        return False
    if _dtype_name(k_dtype) not in FP8_E4M3_DTYPE_NAMES:
        return False
    if _dtype_name(v_dtype) not in FP8_E4M3_DTYPE_NAMES:
        return False
    if _dtype_name(q_dtype) in FP8_E4M3_DTYPE_NAMES:
        return False
    sizes = (q_head_size, k_head_size, v_head_size)
    if any(size != SM90_FP8_MMA_HEAD_SIZE for size in sizes):
        return False
    if not causal and _is_full_window(window_size):
        raise NotImplementedError(
            "SM90 FP8 MMA noncausal full attention has known accuracy errors."
        )
    return True


def is_sm90_fp8_kv_mma(
    q,
    k,
    v,
    causal: bool,
    window_size,
    *,
    device_capability_major: Optional[int] = None,
) -> bool:
    """Tensor-level wrapper; resolves the device capability lazily."""
    if device_capability_major is None:
        try:
            import torch

            device_capability_major = torch.cuda.get_device_capability(q.device)[0]
        except Exception:
            return False
    return is_sm90_fp8_kv_mma_dtypes(
        getattr(q, "dtype", None),
        getattr(k, "dtype", None),
        getattr(v, "dtype", None),
        _head_size(q),
        _head_size(k),
        _head_size(v),
        causal,
        window_size,
        device_capability_major=device_capability_major,
    )


def _is_torch_tensor(value) -> bool:
    return hasattr(value, "dim") and hasattr(value, "dtype")


def _normalize_torch(descale, batch: int, heads: int):
    import torch

    if descale.dim() == 0:
        return descale.to(torch.float32).expand(batch, heads).contiguous()
    if descale.dim() == 1:
        if descale.numel() == 1:
            return descale.to(torch.float32).expand(batch, heads).contiguous()
        if descale.numel() != heads:
            raise ValueError(
                f"descale has {descale.numel()} elements, expected 1 or {heads}"
            )
        return descale.to(torch.float32).reshape(1, heads).expand(batch, heads).contiguous()
    if descale.dim() == 2:
        if tuple(descale.shape) != (batch, heads):
            raise ValueError(
                f"descale shape {tuple(descale.shape)} != ({batch}, {heads})"
            )
        return descale.to(torch.float32).contiguous()
    if descale.dim() == 4 and tuple(descale.shape) == (batch, heads, 1, 1):
        return descale.to(torch.float32).reshape(batch, heads).contiguous()
    raise ValueError(f"unsupported descale shape {tuple(descale.shape)}")


def _normalize_python(descale, batch: int, heads: int):
    if isinstance(descale, (int, float)):
        return [[float(descale)] * heads for _ in range(batch)]
    values = list(descale)
    if not values:
        raise ValueError("descale sequence is empty")
    if isinstance(values[0], (list, tuple)):
        rows = [list(row) for row in values]
        if len(rows) == 1:
            rows = rows * batch
        if len(rows) != batch or any(len(row) != heads for row in rows):
            raise ValueError(f"descale must be ({batch}, {heads})")
        return [[float(x) for x in row] for row in rows]
    if len(values) == 1:
        return [[float(values[0])] * heads for _ in range(batch)]
    if len(values) != heads:
        raise ValueError(f"descale has {len(values)} elements, expected 1 or {heads}")
    row = [float(x) for x in values]
    return [list(row) for _ in range(batch)]


def normalize_kv_descale(descale, *, batch: int, heads: int):
    """Normalize a K/V descale to a contiguous f32 (batch, heads) layout."""
    if descale is None:
        return None
    if _is_torch_tensor(descale):
        return _normalize_torch(descale, batch, heads)
    return _normalize_python(descale, batch, heads)


def select_num_splits(
    *,
    fa_version: int,
    max_seqlen_k: int,
    num_heads: int,
    num_kv_heads: int,
    num_sms: int,
    is_fp8_mma: bool,
) -> int:
    """Resolve the auto split count; 0 means "let the kernel decide"."""
    if fa_version == 3:
        return FA3_SPLIT_CAP
    if not is_fp8_mma:
        return 0
    if max_seqlen_k <= 0 or num_sms <= 0:
        return 1
    splits = max_seqlen_k // FP8_MMA_MIN_SEQLEN_PER_SPLIT
    splits = min(splits, FP8_MMA_MAX_SPLITS)
    # Each split needs at least one CTA per KV head group to fill an SM wave.
    head_groups = max(1, num_heads // max(1, num_kv_heads))
    splits = min(splits, max(1, num_sms // head_groups))
    return max(1, splits)
