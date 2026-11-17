# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for explicit activation-quantization fallback handling.

These tests intentionally avoid importing the ``sglang`` package so they can
run in a minimal environment (no GPU, no compiled kernels, no full dependency
set). The module under test is loaded directly from its file path.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest


_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "python"
    / "sglang"
    / "srt"
    / "layers"
    / "quantization"
    / "activation_fallback.py"
)


def _load_module():
    """Load the module under test without requiring a full sglang install.

    Prefer a normal import so the test exercises the real package layout when
    the environment is healthy; fall back to loading the file directly so the
    suite still runs in minimal environments where ``pip install -e python``
    has not completed.
    """
    try:
        from sglang.srt.layers.quantization import activation_fallback as module

        return module
    except Exception:
        pass

    spec = importlib.util.spec_from_file_location(
        "_activation_fallback_under_test", _MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_af = _load_module()

check_activation_quant_fallback = _af.check_activation_quant_fallback
executed_activation_key = _af.executed_activation_key
is_strict_quant_scheme = _af.is_strict_quant_scheme


# ---------------------------------------------------------------------------
# Fake kernels mirroring the real backends' declarations.
# ---------------------------------------------------------------------------


class _HonoringKernel:
    """Kernel that honors whatever activation scheme is requested."""


class _WeightOnlyKernel:
    """Kernel that always runs unquantized activations (e.g. Marlin)."""

    @staticmethod
    def executed_activation_key(weight_key, activation_key):
        return None


class _DeclaredKernel:
    """Kernel that declares a specific executed activation key."""

    @staticmethod
    def executed_activation_key(weight_key, activation_key):
        return "executed"


class _HummingKernel:
    """Humming runs the input schema its weights were converted with."""

    @staticmethod
    def executed_activation_key(weight_key, activation_key):
        if weight_key in ("kInt4Static", "kMxfp4Static"):
            return activation_key
        return None


class _Mxfp4Kernel:
    """MXFP4 weight-only conversion runs unquantized activations."""

    @staticmethod
    def executed_activation_key(weight_key, activation_key):
        return None


class _Fp8Kernel:
    """FP8 W8A8 kernel honors the requested activation scheme."""

    @staticmethod
    def executed_activation_key(weight_key, activation_key):
        return activation_key


# ---------------------------------------------------------------------------
# Strict-mode env var handling.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["", "0"])
def test_strict_disabled(monkeypatch, value):
    monkeypatch.setenv("VLLM_STRICT_QUANT_SCHEME", value)
    assert is_strict_quant_scheme() is False


@pytest.mark.parametrize("value", ["1", "true", "yes"])
def test_strict_enabled(monkeypatch, value):
    monkeypatch.setenv("VLLM_STRICT_QUANT_SCHEME", value)
    assert is_strict_quant_scheme() is True


def test_strict_unset(monkeypatch):
    monkeypatch.delenv("VLLM_STRICT_QUANT_SCHEME", raising=False)
    assert is_strict_quant_scheme() is False


# ---------------------------------------------------------------------------
# executed_activation_key dispatch.
# ---------------------------------------------------------------------------


def test_executed_activation_key_default_honors_request():
    assert executed_activation_key(_HonoringKernel, "w", "a") == "a"


def test_executed_activation_key_declared():
    assert executed_activation_key(_WeightOnlyKernel, "w", "a") is None
    assert executed_activation_key(_DeclaredKernel, "w", "a") == "executed"


@pytest.mark.parametrize(
    "kernel,weight_key,activation_key,expected",
    [
        (_HummingKernel, "kInt4Static", "a", "a"),
        (_HummingKernel, "kMxfp4Static", "a", "a"),
        (_HummingKernel, "kFp8Static128BlockSym", "a", None),
        (_Mxfp4Kernel, "kMxfp4Static", "a", None),
        (_Fp8Kernel, "kFp8Static128BlockSym", "a", "a"),
    ],
)
def test_executed_activation_key_per_backend(
    kernel, weight_key, activation_key, expected
):
    assert executed_activation_key(kernel, weight_key, activation_key) == expected


# ---------------------------------------------------------------------------
# check_activation_quant_fallback helper.
# ---------------------------------------------------------------------------


def test_check_no_fallback_when_keys_match():
    assert check_activation_quant_fallback("k", "a", "a") is None


def test_check_no_fallback_when_nothing_requested():
    assert check_activation_quant_fallback("k", None, None) is None


def test_check_reports_fallback_in_default_mode(monkeypatch):
    monkeypatch.delenv("VLLM_STRICT_QUANT_SCHEME", raising=False)
    reason = check_activation_quant_fallback("k", "a", None)
    assert reason is not None
    assert "k" in reason


def test_check_raises_in_strict_mode(monkeypatch):
    monkeypatch.setenv("VLLM_STRICT_QUANT_SCHEME", "1")
    with pytest.raises(ValueError, match="VLLM_STRICT_QUANT_SCHEME"):
        check_activation_quant_fallback("k", "a", None)


def test_check_strict_error_mentions_mismatch(monkeypatch):
    monkeypatch.setenv("VLLM_STRICT_QUANT_SCHEME", "1")
    with pytest.raises(ValueError) as excinfo:
        check_activation_quant_fallback("k", "requested", "executed")
    message = str(excinfo.value)
    assert "requested" in message
    assert "executed" in message


def test_check_strict_does_not_raise_when_honored(monkeypatch):
    monkeypatch.setenv("VLLM_STRICT_QUANT_SCHEME", "1")
    assert check_activation_quant_fallback("k", "a", "a") is None


# ---------------------------------------------------------------------------
# Backend selection paths: default mode picks fallback, strict mode rejects.
# ---------------------------------------------------------------------------


def _select(kernels, requested_key, strict):
    """Minimal stand-in for the backend selection loop.

    Mirrors the upstream contract: iterate kernels in priority order, skip any
    kernel whose executed activation key does not honor the requested scheme
    when strict mode is on, and otherwise return the first kernel that can run.
    """
    if strict:
        os.environ["VLLM_STRICT_QUANT_SCHEME"] = "1"
    else:
        os.environ.pop("VLLM_STRICT_QUANT_SCHEME", None)
    try:
        for kernel in kernels:
            executed = executed_activation_key(kernel, "w", requested_key)
            reason = check_activation_quant_fallback(
                kernel.__name__, requested_key, executed
            )
            if reason is None:
                return kernel
        raise ValueError(
            "no kernel honors the requested activation quantization"
        )
    finally:
        os.environ.pop("VLLM_STRICT_QUANT_SCHEME", None)


@pytest.mark.parametrize(
    "kernels,requested_key,expected",
    [
        ([_WeightOnlyKernel], "a", _WeightOnlyKernel),
        ([_Mxfp4Kernel], "a", _Mxfp4Kernel),
        ([_HummingKernel], "a", _HummingKernel),
        ([_Fp8Kernel], "a", _Fp8Kernel),
        ([_WeightOnlyKernel, _Fp8Kernel], "a", _WeightOnlyKernel),
    ],
)
def test_default_mode_selects_fallback(kernels, requested_key, expected):
    assert _select(kernels, requested_key, strict=False) is expected


@pytest.mark.parametrize(
    "kernels,requested_key",
    [
        ([_WeightOnlyKernel], "a"),
        ([_Mxfp4Kernel], "a"),
        ([_HummingKernel], "a"),
    ],
)
def test_strict_mode_rejects_fallback(kernels, requested_key):
    with pytest.raises(ValueError, match="VLLM_STRICT_QUANT_SCHEME"):
        _select(kernels, requested_key, strict=True)


def test_strict_mode_falls_through_to_honoring_kernel():
    # Priority order is preserved: a later kernel that honors the scheme is
    # still selected instead of failing startup.
    assert (
        _select([_WeightOnlyKernel, _Fp8Kernel], "a", strict=True) is _Fp8Kernel
    )


def test_strict_mode_accepts_honoring_kernel():
    assert _select([_Fp8Kernel], "a", strict=True) is _Fp8Kernel


# ---------------------------------------------------------------------------
# MoE runner declarations (base / humming / marlin).
# ---------------------------------------------------------------------------


class _BaseRunner:
    """Default runner: assumes the requested scheme is honored."""

    @staticmethod
    def executed_activation_key(weight_key, activation_key):
        return activation_key


class _HummingRunner:
    """Humming runner: honors the scheme only when an input config is set."""

    def __init__(self, has_input_config):
        self._has_input_config = has_input_config

    def executed_activation_key(self, weight_key, activation_key):
        if self._has_input_config:
            return activation_key
        return None


class _MarlinRunner:
    """Marlin runner: weight-only, activations always run in 16-bit."""

    @staticmethod
    def executed_activation_key(weight_key, activation_key):
        return None


@pytest.mark.parametrize(
    "runner,expected",
    [
        (_BaseRunner, "a"),
        (_MarlinRunner, None),
    ],
)
def test_moe_runner_executed_activation_key(runner, expected):
    assert executed_activation_key(runner, "w", "a") == expected


def test_humming_runner_with_input_config_honors_scheme():
    runner = _HummingRunner(has_input_config=True)
    assert executed_activation_key(runner, "w", "a") == "a"


def test_humming_runner_without_input_config_falls_back():
    runner = _HummingRunner(has_input_config=False)
    assert executed_activation_key(runner, "w", "a") is None


def test_marlin_runner_rejected_under_strict(monkeypatch):
    monkeypatch.setenv("VLLM_STRICT_QUANT_SCHEME", "1")
    executed = executed_activation_key(_MarlinRunner, "w", "a")
    with pytest.raises(ValueError, match="VLLM_STRICT_QUANT_SCHEME"):
        check_activation_quant_fallback("MarlinRunner", "a", executed)


def test_marlin_runner_warns_in_default_mode(monkeypatch):
    monkeypatch.delenv("VLLM_STRICT_QUANT_SCHEME", raising=False)
    executed = executed_activation_key(_MarlinRunner, "w", "a")
    reason = check_activation_quant_fallback("MarlinRunner", "a", executed)
    assert reason is not None
    assert "MarlinRunner" in reason
