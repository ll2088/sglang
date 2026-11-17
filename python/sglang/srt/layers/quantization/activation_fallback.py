# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) vLLM project
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

# NOTE: this module must stay importable with only the standard library.
# It is loaded directly from its source file by tests/quantization/conftest.py
# so the suite can run with pytest alone (no sglang import, no GPU, no
# compiled kernels). Do not add sglang imports here.
logger = logging.getLogger(__name__)

# Messages already emitted by ``_warn_once``; used when the stdlib logger is
# in play (no ``warning_once`` extension).
_WARNED_MESSAGES: set[str] = set()

__all__ = [
    "UNQUANTIZED_ACTIVATION",
    "is_strict_quant_scheme",
    "executed_activation_key",
    "check_activation_quant_fallback",
    "check_runner_activation_fallback",
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

    The environment is consulted lazily (never cached at import time) so that
    the module stays importable without any sglang dependency and so that the
    strict-mode switch can be flipped at runtime.
    """

    value = os.environ.get(_STRICT_ENV_VAR, "")
    return value not in ("", "0")


def _warn_once(message: str, *args: Any) -> None:
    """Emit a warning at most once per unique message.

    ``logging.Logger.warning_once`` is a vLLM/sglang extension that is not
    available on the stdlib logger. This helper keeps the module importable
    with only the standard library while preserving the once-per-kernel
    behaviour the upstream change relies on.
    """

    warn_once = getattr(logger, "warning_once", None)
    if warn_once is not None:
        warn_once(message, *args)
        return

    key = message % args if args else message
    if key in _WARNED_MESSAGES:
        return
    _WARNED_MESSAGES.add(key)
    logger.warning(message, *args)


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
            f"{reason}; VLLM_STRICT_QUANT_SCHEME=1 forbids this fallback "
            f"(requested={requested_key!r}, executed={executed_key!r})"
        )

    _warn_once(
        "%s; falling back to unquantized activations. Set "
        "VLLM_STRICT_QUANT_SCHEME=1 to reject this kernel.",
        reason,
    )
    return reason


def check_runner_activation_fallback(
    runner_name: str,
    weight_key: Optional[Any],
    activation_key: Optional[Any],
    runner: Any,
) -> Optional[str]:
    """Unified fallback check for MoE runners.

    Resolves the activation quantization ``runner`` actually executes via
    :func:`executed_activation_key` and compares it with the requested
    ``activation_key`` through :func:`check_activation_quant_fallback`. This is
    the single entry point MoE runner selection should use so that no silent
    fallback branch is left behind.
    """

    executed_key = executed_activation_key(runner, weight_key, activation_key)
    return check_activation_quant_fallback(runner_name, activation_key, executed_key)
