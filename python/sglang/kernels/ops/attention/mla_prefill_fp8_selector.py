"""Opt-in FlyDSL FP8 MLA prefill backend selection (gfx950).

Default is off: ``select_mla_prefill_backend`` returns ``None`` unless the
caller explicitly opts in, so existing prefill paths are untouched. The
resolver only imports the FlyDSL kernel when the opt-in is enabled and the
runtime advertises support, keeping default imports dependency-free.
"""

from __future__ import annotations

from typing import Optional

BACKEND_NAME = "ROCM_AITER_FLYDSL_FP8"


def _flydsl_fp8_supported() -> bool:
    try:
        from aiter.ops.flydsl import flydsl_flash_attn_fp8_supported  # noqa: F401
    except ImportError:
        return False
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        from aiter.jit.utils.chip_info import get_gfx

        return get_gfx() == "gfx950"
    except Exception:  # noqa: BLE001
        return False


def select_mla_prefill_backend(
    requested: Optional[str] = None,
) -> Optional[str]:
    """Return the backend name to use, or ``None`` to keep the existing path."""
    if requested != BACKEND_NAME:
        return None
    if not _flydsl_fp8_supported():
        return None
    return BACKEND_NAME


def resolve_mla_prefill_kernel():
    """Import and return the FlyDSL FP8 prefill entry point, or ``None``."""
    if select_mla_prefill_backend(BACKEND_NAME) is None:
        return None
    from aiter.ops.flydsl import flydsl_flash_attn_fp8_func

    return flydsl_flash_attn_fp8_func


__all__ = [
    "BACKEND_NAME",
    "select_mla_prefill_backend",
    "resolve_mla_prefill_kernel",
]
