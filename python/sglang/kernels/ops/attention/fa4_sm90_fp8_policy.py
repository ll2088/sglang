# Copyright (c) 2026, SGLang Team.
"""Pure routing policy for the FA4 SM90 d512 FP8-MMA path.

Kept torch-free at import time so the routing tests and the CI smoke check can
run on a CPU-only runner without the compiled FA4 extension. Tensor inputs are
only inspected through ``dtype``/``shape`` attributes, never imported here.
"""

from typing import Optional, Sequence

SM90_FP8_MMA_HEAD_SIZE = 512
FA3_SPLIT_CAP = 32
FP8_MMA_MAX_SPLITS = 4
FP8_MMA_MIN_TILES_PER_SPLIT = 2
FP8_MMA_TILE_N = 64


def _dtype_name(dtype) -> str:
    return getattr(dtype, "__name__", str(dtype)).replace("torch.", "")


def _is_fp8_e4m3(dtype) -> bool:
    return _dtype_name(dtype) in ("float8_e4m3fn", "float8_e4m3fnuz")


def _is_fp16_or_bf16(dtype) -> bool:
    return _dtype_name(dtype) in ("float16", "bfloat16")


def _head_size(t) -> Optional[int]:
    shape = getattr(t, "shape", None)
    return None if shape is None else int(shape[-1])


def _is_local(window_size) -> bool:
    if window_size is None:
        return False
    left, right = window_size
    return left is not None or right is not None


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
    device_capability_major: int,
) -> bool:
    """Return whether the call qualifies for the SM90 d512 FP8-MMA kernel."""
    if device_capability_major != 9:
        return False
    if not _is_fp16_or_bf16(q_dtype):
        return False
    if not (_is_fp8_e4m3(k_dtype) and _is_fp8_e4m3(v_dtype)):
        return False
    if not (
        q_head_size == k_head_size == v_head_size == SM90_FP8_MMA_HEAD_SIZE
    ):
        return False
    if not causal and not _is_local(window_size):
        # Noncausal full attention has known accuracy errors on this kernel.
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
    device_capability_major: int,
) -> bool:
    """Tensor-level wrapper around :func:`is_sm90_fp8_kv_mma_dtypes`."""
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


def _broadcast_rows(values: Sequence, batch: int, heads: int) -> list:
    rows = list(values)
    if len(rows) == 1:
        rows = rows * batch
    if len(rows) != batch:
        raise ValueError(
            f"descale batch dimension {len(rows)} does not match batch={batch}"
        )
    out = []
    for row in rows:
        if isinstance(row, (list, tuple)):
            row = list(row)
            if len(row) == 1:
                row = row * heads
            if len(row) != heads:
                raise ValueError(
                    f"descale head dimension {len(row)} does not match heads={heads}"
                )
            out.append(row)
        else:
            out.append([row] * heads)
    return out


def normalize_kv_descale(descale, *, batch: int, heads: int):
    """Normalize a K/V descale to a contiguous f32 ``(batch, heads)`` layout.

    Accepts ``None``, a scalar, a per-head vector, a ``(batch, heads)`` tensor,
    or a trailing-singleton broadcast layout. Torch tensors are returned as
    contiguous f32 tensors; plain Python sequences are returned as nested
    lists so the policy stays importable without torch.
    """
    if descale is None:
        return None
    if hasattr(descale, "dtype") and hasattr(descale, "shape"):
        import torch

        t = descale
        if t.numel() == 1:
            t = t.reshape(1, 1)
        else:
            t = t.reshape(-1) if t.ndim == 1 else t.reshape(t.shape[0], -1)
            if t.shape[0] == 1 and batch > 1:
                t = t.expand(batch, t.shape[1])
            if t.shape[1] == 1 and heads > 1:
                t = t.expand(t.shape[0], heads)
        if tuple(t.shape) != (batch, heads):
            raise ValueError(
                f"descale shape {tuple(t.shape)} does not match ({batch}, {heads})"
            )
        return t.to(dtype=torch.float32).contiguous()
    if isinstance(descale, (int, float)):
        return descale
    return _broadcast_rows(descale, batch, heads)


def select_num_splits(
    *,
    fa_version: int,
    max_seqlen_k: int,
    num_heads: int,
    num_kv_heads: int,
    num_sms: int,
    is_fp8_mma: bool,
) -> int:
    """Pick the SplitKV count; FA3 keeps its cap, FA4 FP8-MMA is bounded."""
    if fa_version == 3:
        return FA3_SPLIT_CAP
    if not is_fp8_mma:
        return 0
    if max_seqlen_k <= 0 or num_sms <= 0:
        return 1
    num_n_blocks = (max_seqlen_k + FP8_MMA_TILE_N - 1) // FP8_MMA_TILE_N
    by_tiles = num_n_blocks // FP8_MMA_MIN_TILES_PER_SPLIT
    by_sms = max(1, num_sms // max(1, num_heads // max(1, num_kv_heads)))
    return max(1, min(FP8_MMA_MAX_SPLITS, by_tiles, by_sms))
