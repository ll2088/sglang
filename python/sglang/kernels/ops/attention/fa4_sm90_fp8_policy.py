# Copyright (c) 2026, SGLang Team.
"""Pure routing policy for the FA4 SM90 d512 FP8-MMA path.

Kept free of torch.cuda / cute imports so it can be unit-tested on CPU
without a GPU or a compiled flash-attention extension.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    import torch

# NOTE: This module must stay importable without the sglang package or a
# compiled flash-attention extension so routing can be unit-tested on CPU.
# torch is imported lazily inside the helpers that need it, so importing this
# module never pulls in CUDA initialization.

SM90_FP8_MMA_HEAD_DIM = 512
SM90_FP8_MMA_ARCH_MAJOR = 9
FA3_SPLIT_CAP = 32
SPLIT_KV_TILE_N = 128


def is_sm90_fp8_kv_mma(
    q: "torch.Tensor",
    k: "torch.Tensor",
    v: "torch.Tensor",
    causal: bool,
    window_size,
    *,
    device_capability_major: Optional[int] = None,
) -> bool:
    """Return whether the SM90 d512 FP8-MMA path applies to this call.

    The kernel accepts native FP16/BF16 Q with paged E4M3 K/V and folds the
    K/V descales in-kernel. It is only valid for equal head dims of 512 on
    SM90 and rejects noncausal full attention, which has known accuracy
    issues.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover - CPU-only policy import
        return False

    if k.dtype != torch.float8_e4m3fn:
        return False
    if not (q.shape[-1] == k.shape[-1] == v.shape[-1] == SM90_FP8_MMA_HEAD_DIM):
        return False
    if device_capability_major is None:
        device_capability_major = torch.cuda.get_device_capability()[0]
    if device_capability_major != SM90_FP8_MMA_ARCH_MAJOR:
        return False
    if not causal and window_size == (None, None):
        raise NotImplementedError(
            "SM90 FP8 MMA noncausal full attention has known accuracy errors."
        )
    return True


def normalize_kv_descale(
    descale: Optional["torch.Tensor"],
    *,
    batch: int,
    heads: int,
) -> Optional["torch.Tensor"]:
    """Normalize a K/V descale tensor to the (batch, heads) f32 layout.

    Accepts scalars, (heads,), (batch, heads) and (batch, heads, 1, 1)
    layouts so callers can pass either FA3-style or FA4-style descales.
    """
    if descale is None:
        return None
    try:
        import torch
    except ImportError:  # pragma: no cover - CPU-only policy import
        return None

    t = torch.as_tensor(descale, dtype=torch.float32)
    if t.ndim == 0:
        return t.expand(batch, heads).contiguous()
    if t.ndim == 1:
        if t.shape[0] != heads:
            raise ValueError(f"expected {heads} heads, got {t.shape[0]}")
        return t.unsqueeze(0).expand(batch, heads).contiguous()
    if t.ndim == 2:
        if tuple(t.shape) != (batch, heads):
            raise ValueError(f"expected {(batch, heads)}, got {tuple(t.shape)}")
        return t.contiguous()
    if t.ndim == 4:
        if t.shape[0] != batch or t.shape[1] != heads:
            raise ValueError(
                f"expected {(batch, heads, 1, 1)}, got {tuple(t.shape)}"
            )
        if t.shape[2] != 1 or t.shape[3] != 1:
            raise ValueError(f"expected trailing (1, 1), got {tuple(t.shape[2:])}")
        return t.reshape(batch, heads).contiguous()
    raise ValueError(f"unsupported descale layout: {tuple(t.shape)}")


def select_num_splits(
    *,
    fa_version: int,
    max_seqlen_k: int,
    num_heads: int,
    num_kv_heads: int,
    num_sms: int,
    is_fp8_mma: bool,
    cap: int = FA3_SPLIT_CAP,
) -> int:
    """Mirror the FA4 full-attention split auto-selection policy.

    FA4 full-attention layers (including the SM90 d512 FP8-MMA path) are
    allowed to pick splits automatically without the CUDA-graph cap of 32;
    FA3 keeps the cap. Returns 0 for "auto" when the caller should let the
    kernel decide, otherwise a concrete split count.
    """
    if fa_version == 3:
        return cap
    if not is_fp8_mma:
        return 0
    if max_seqlen_k <= 0 or num_sms <= 0:
        return 1
    if num_heads <= 0:
        return 1
    qhead_per_kvhead = max(1, num_heads // max(1, num_kv_heads))
    # One CTA per (kv head, split); aim for at least two KV tiles per split.
    tiles = max(1, max_seqlen_k // SPLIT_KV_TILE_N)
    target = max(1, num_sms // max(1, num_kv_heads * qhead_per_kvhead))
    return max(1, min(target, tiles))
