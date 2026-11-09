"""Opt-in FlyDSL FP8 MLA prefill backend selection (gfx950 / ROCm).

Mirrors the upstream ``ROCM_AITER_FLYDSL_FP8`` idea in the smallest form that
this repository can carry without a new runtime dependency: the backend is
never auto-selected, availability is decided by an explicit opt-in flag plus
an optional AITER capability probe, and anything the kernel cannot take falls
back to the existing BF16 prefill path.

This module is import-safe on CPU-only / non-ROCm hosts: it never imports
``torch`` or ``aiter`` at module import time.
"""

from __future__ import annotations

import os
from typing import Optional

# Name used in the attention config / env var, matching upstream.
BACKEND_NAME = "ROCM_AITER_FLYDSL_FP8"

# Env var that turns the backend on. Unset / empty / "0" keeps it off.
OPT_IN_ENV = "SGLANG_MLA_PREFILL_BACKEND"

# FlyDSL FP8 attention packs flattened Q/K/V/O sizes as int32.
FLYDSL_FP8_MAX_NUMEL = 2**31


def _opt_in_backend() -> Optional[str]:
    value = os.environ.get(OPT_IN_ENV, "").strip()
    return value or None


def _aiter_flydsl_fp8_available() -> bool:
    """True only if AITER ships the FlyDSL FP8 attention entry points.

    Imported lazily so that a missing optional dependency is a normal
    ``False`` rather than an import error.
    """
    try:
        from aiter.ops.flydsl import (  # noqa: F401
            flydsl_flash_attn_fp8_func,
            flydsl_flash_attn_fp8_supported,
        )
    except ImportError:
        return False
    return True


def _on_gfx950() -> bool:
    try:
        from aiter.jit.utils.chip_info import get_gfx

        return get_gfx() == "gfx950"
    except Exception:  # noqa: BLE001 - any probe failure means "not supported"
        return False


def select_mla_prefill_backend(
    requested: Optional[str] = None,
    *,
    numel: int = 0,
) -> str:
    """Resolve the MLA prefill backend name for this process.

    ``requested`` (e.g. from an attention config) wins over the env var. The
    FlyDSL FP8 backend is returned only when it was explicitly requested, the
    host is gfx950, AITER exposes the FP8 entry points, and the input is
    non-empty and below the int32 packing limit. Otherwise the caller keeps
    its existing BF16 prefill backend.
    """
    wanted = (requested or "").strip() or _opt_in_backend()
    if wanted != BACKEND_NAME:
        return ""
    if numel <= 0 or numel >= FLYDSL_FP8_MAX_NUMEL:
        return ""
    if not _on_gfx950() or not _aiter_flydsl_fp8_available():
        return ""
    return BACKEND_NAME


def resolve_mla_prefill_kernel(requested: Optional[str] = None, *, numel: int = 0):
    """Return the callable for the selected backend, or ``None`` to fall back.

    Kept as a thin indirection so call sites do not import AITER themselves.
    """
    if select_mla_prefill_backend(requested, numel=numel) != BACKEND_NAME:
        return None
    from aiter.ops.flydsl import flydsl_flash_attn_fp8_func

    return flydsl_flash_attn_fp8_func


__all__ = [
    "BACKEND_NAME",
    "FLYDSL_FP8_MAX_NUMEL",
    "OPT_IN_ENV",
    "resolve_mla_prefill_kernel",
    "select_mla_prefill_backend",
]
