"""Capability probe for the opt-in FlyDSL FP8 MLA prefill backend.

Pure-Python predicates so the selection logic can be unit-tested without a
GPU, AITER or FlyDSL install. The backend is opt-in: it is only selected when
the user enables it, the device is gfx950, and FlyDSL FP8 attention is
available. Otherwise callers fall back to the default prefill path.
"""

from __future__ import annotations

from typing import Optional

__all__ = [
    "should_use_flydsl_fp8_prefill",
    "flydsl_fp8_fallback_reason",
]

_REQUIRED_GFX = "gfx950"


def should_use_flydsl_fp8_prefill(
    *, enabled: bool, gfx: Optional[str], flydsl_available: bool
) -> bool:
    """True only when the opt-in backend is enabled and fully supported."""
    return (
        bool(enabled)
        and gfx == _REQUIRED_GFX
        and bool(flydsl_available)
    )


def flydsl_fp8_fallback_reason(
    *, enabled: bool, gfx: Optional[str], flydsl_available: bool
) -> Optional[str]:
    """Human-readable reason the opt-in backend is not used, or ``None``."""
    if not enabled:
        return "FlyDSL FP8 prefill is opt-in and not enabled"
    if gfx != _REQUIRED_GFX:
        return f"FlyDSL FP8 prefill requires {_REQUIRED_GFX}, got {gfx!r}"
    if not flydsl_available:
        return "FlyDSL FP8 attention is not available in this AITER build"
    return None
