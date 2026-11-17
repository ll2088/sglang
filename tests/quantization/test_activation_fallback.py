# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for explicit activation-quantization fallback handling."""

from __future__ import annotations

import pytest

from sglang.srt.layers.quantization.activation_fallback import (
    check_activation_quant_fallback,
    executed_activation_key,
    is_strict_quant_scheme,
)


class _HonoringKernel:
    """Kernel that honors whatever activation scheme is requested."""


class _WeightOnlyKernel:
    """Kernel that always runs unquantized activations."""

    @staticmethod
    def executed_activation_key(weight_key, activation_key):
        return None


class _DeclaredKernel:
    """Kernel that declares a specific executed activation key."""

    @staticmethod
    def executed_activation_key(weight_key, activation_key):
        return "executed"


@pytest.mark.parametrize("value", ["", "0"])
def test_strict_disabled(monkeypatch, value):
    monkeypatch.setenv("VLLM_STRICT_QUANT_SCHEME", value)
    assert is_strict_quant_scheme() is False


@pytest.mark.parametrize("value", ["1", "true", "yes"])
def test_strict_enabled(monkeypatch, value):
    monkeypatch.setenv("VLLM_STRICT_QUANT_SCHEME", value)
    assert is_strict_quant_scheme() is True


def test_executed_activation_key_default_honors_request():
    assert executed_activation_key(_HonoringKernel, "w", "a") == "a"


def test_executed_activation_key_declared():
    assert executed_activation_key(_WeightOnlyKernel, "w", "a") is None
    assert executed_activation_key(_DeclaredKernel, "w", "a") == "executed"


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


def test_check_strict_does_not_raise_when_honored(monkeypatch):
    monkeypatch.setenv("VLLM_STRICT_QUANT_SCHEME", "1")
    assert check_activation_quant_fallback("k", "a", "a") is None
