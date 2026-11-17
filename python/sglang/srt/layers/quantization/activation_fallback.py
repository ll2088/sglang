# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit activation-quantization fallback handling.

Weight-only kernels (Marlin, Humming weight-only conversions, MXFP8 emulation,
...) accept whatever activation scheme a checkpoint requests but run
activations in 16-bit, so an FP8 W8A8 or NVFP4 W4A4 layer quietly becomes
W8A16/W4A16 when native kernels are unavailable.

This module makes that fallback visible (one warning per kernel) and adds
``VLLM_STRICT_QUANT_SCHEME=1`` to forbid it. Each kernel declares the
activation quantization it actually executes via ``executed_activation_key``;
selection compares it with the requested key through
``check_activation_quant_fallback``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "UNQUANTIZED_ACTIVATION",
    "is_strict_quant_scheme",
    "executed_activation_key",
    "check_activation_quant_fallback",
]

# Sentinel used to distinguish "no activation quantization requested" from
# "the kernel runs unquantized activations".
UNQUANTIZED_ACTIVATION = None

_STRICT_ENV_VAR = "VLLM_STRICT_QUANT_SCHEME"


def is_strict_quant_scheme() -> bool:
    """Return True when ``VLLM_STRICT_QUANT_SCHEME`` forbids silent fallbacks.

    The variable is read on every call so tests can toggle it via monkeypatch
    without re-importing the module. Any value other than ``"0"``/``""`` is
    treated as enabled, matching the upstream semantics.
    """

    value = os.environ.get(_STRICT_ENV_VAR, "")
    return value not in ("", "0")


def executed_activation_key(
    kernel: Any,
    weight_key: Optional[Any],
    activation_key: Optional[Any],
) -> Optional[Any]:
    """Return the activation quantization ``kernel`` actually executes.

    Kernels may override this by defining a static/class method named
    ``executed_activation_key`` taking ``(weight_key, activation_key)``. When
    they do not, the requested ``activation_key`` is assumed to be honored.
    """

    declared = getattr(kernel, "executed_activation_key", None)
    if declared is None:
        return activation_key
    return declared(weight_key, activation_key)


def check_activation_quant_fallback(
    kernel_name: str,
    requested_key: Optional[Any],
    executed_key: Optional[Any],
) -> Optional[str]:
    """Compare the requested and executed activation quantization keys.

    Returns ``None`` when the kernel honors the requested scheme. Otherwise
    returns a human-readable reason describing the fallback. In strict mode the
    caller is expected to turn that reason into a startup error; in default mode
    a once-per-kernel warning is emitted here.
    """

    if requested_key is None or requested_key == executed_key:
        return None

    reason = (
        f"kernel {kernel_name} runs activation quantization "
        f"{executed_key!r} but {requested_key!r} was requested"
    )

    if is_strict_quant_scheme():
        raise ValueError(
            f"{reason}; VLLM_STRICT_QUANT_SCHEME=1 forbids this fallback"
        )

    logger.warning_once(
        "%s; falling back to unquantized activations. Set "
        "VLLM_STRICT_QUANT_SCHEME=1 to reject this kernel.",
        reason,
    )
    return reason
