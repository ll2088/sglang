# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for explicit activation-quantization fallback handling.

These tests exercise ``activation_fallback.py`` in isolation. The module is
loaded from its file path by ``conftest.py`` so the suite runs with only pytest
installed (no sglang import, no GPU, no compiled kernels).
"""

from __future__ import annotations

import logging

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Kernel:
    """Kernel that honors whatever activation scheme is requested."""

    @staticmethod
    def executed_activation_key(weight_key, activation_key):
        return activation_key


class _WeightOnlyKernel:
    """Kernel that always runs unquantized activations (e.g. Marlin)."""

    @staticmethod
    def executed_activation_key(weight_key, activation_key):
        return None


class _NoDeclarationKernel:
    """Kernel without an ``executed_activation_key`` declaration."""


class _HummingKernel:
    """Humming runs the input schema its weights were converted with."""

    def __init__(self, has_input_quant_config: bool):
        self._has_input_quant_config = has_input_quant_config

    def executed_activation_key(self, weight_key, activation_key):
        if self._has_input_quant_config:
            return activation_key
        return None


# ---------------------------------------------------------------------------
# is_strict_quant_scheme
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, False),
        ("", False),
        ("0", False),
        ("1", True),
        ("true", True),
        ("yes", True),
    ],
)
def test_is_strict_quant_scheme(monkeypatch, activation_fallback, value, expected):
    if value is None:
        monkeypatch.delenv("VLLM_STRICT_QUANT_SCHEME", raising=False)
    else:
        monkeypatch.setenv("VLLM_STRICT_QUANT_SCHEME", value)
    assert activation_fallback.is_strict_quant_scheme() is expected


# ---------------------------------------------------------------------------
# executed_activation_key
# ---------------------------------------------------------------------------


def test_executed_activation_key_defaults_to_requested(activation_fallback):
    assert (
        activation_fallback.executed_activation_key(
            _NoDeclarationKernel, "w", "a"
        )
        == "a"
    )


def test_executed_activation_key_uses_declaration(activation_fallback):
    assert (
        activation_fallback.executed_activation_key(_Kernel, "w", "a") == "a"
    )
    assert (
        activation_fallback.executed_activation_key(_WeightOnlyKernel, "w", "a")
        is None
    )


# ---------------------------------------------------------------------------
# check_activation_quant_fallback
# ---------------------------------------------------------------------------


def test_no_fallback_when_requested_is_none(activation_fallback):
    assert (
        activation_fallback.check_activation_quant_fallback(
            "k", None, "anything"
        )
        is None
    )


def test_no_fallback_when_keys_match(activation_fallback):
    assert (
        activation_fallback.check_activation_quant_fallback("k", "a", "a")
        is None
    )


def test_fallback_returns_reason_in_default_mode(
    monkeypatch, activation_fallback, caplog
):
    monkeypatch.delenv("VLLM_STRICT_QUANT_SCHEME", raising=False)
    with caplog.at_level(logging.WARNING):
        reason = activation_fallback.check_activation_quant_fallback(
            "MarlinExperts", "fp8", None
        )
    assert reason is not None
    assert "MarlinExperts" in reason
    assert "fp8" in reason


def test_fallback_raises_in_strict_mode(monkeypatch, activation_fallback):
    monkeypatch.setenv("VLLM_STRICT_QUANT_SCHEME", "1")
    with pytest.raises(ValueError, match="VLLM_STRICT_QUANT_SCHEME"):
        activation_fallback.check_activation_quant_fallback(
            "MarlinExperts", "fp8", None
        )


def test_fallback_error_message_contains_mismatch(monkeypatch, activation_fallback):
    monkeypatch.setenv("VLLM_STRICT_QUANT_SCHEME", "1")
    with pytest.raises(ValueError) as excinfo:
        activation_fallback.check_activation_quant_fallback(
            "MarlinExperts", "fp8", None
        )
    message = str(excinfo.value)
    assert "MarlinExperts" in message
    assert "fp8" in message
    assert "VLLM_STRICT_QUANT_SCHEME" in message


# ---------------------------------------------------------------------------
# check_runner_activation_fallback (MoE runner path)
# ---------------------------------------------------------------------------


def test_runner_fallback_default_mode(monkeypatch, activation_fallback):
    monkeypatch.delenv("VLLM_STRICT_QUANT_SCHEME", raising=False)
    reason = activation_fallback.check_runner_activation_fallback(
        "MarlinMoeQuantInfo", "w", "fp8", _WeightOnlyKernel
    )
    assert reason is not None
    assert "MarlinMoeQuantInfo" in reason


def test_runner_fallback_strict_mode(monkeypatch, activation_fallback):
    monkeypatch.setenv("VLLM_STRICT_QUANT_SCHEME", "1")
    with pytest.raises(ValueError, match="VLLM_STRICT_QUANT_SCHEME"):
        activation_fallback.check_runner_activation_fallback(
            "MarlinMoeQuantInfo", "w", "fp8", _WeightOnlyKernel
        )


def test_runner_no_fallback_when_honored(activation_fallback):
    assert (
        activation_fallback.check_runner_activation_fallback(
            "Kernel", "w", "fp8", _Kernel
        )
        is None
    )


# ---------------------------------------------------------------------------
# Backend-specific declarations (humming / marlin / mxfp4 / fp8)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strict", [False, True])
def test_marlin_weight_only_fallback(monkeypatch, activation_fallback, strict):
    monkeypatch.setenv("VLLM_STRICT_QUANT_SCHEME", str(int(strict)))
    if strict:
        with pytest.raises(ValueError, match="VLLM_STRICT_QUANT_SCHEME"):
            activation_fallback.check_runner_activation_fallback(
                "MarlinMoeQuantInfo", "w", "fp8", _WeightOnlyKernel
            )
    else:
        assert (
            activation_fallback.check_runner_activation_fallback(
                "MarlinMoeQuantInfo", "w", "fp8", _WeightOnlyKernel
            )
            is not None
        )


@pytest.mark.parametrize("strict", [False, True])
def test_humming_without_input_quant_config_falls_back(
    monkeypatch, activation_fallback, strict
):
    monkeypatch.setenv("VLLM_STRICT_QUANT_SCHEME", str(int(strict)))
    kernel = _HummingKernel(has_input_quant_config=False)
    if strict:
        with pytest.raises(ValueError, match="VLLM_STRICT_QUANT_SCHEME"):
            activation_fallback.check_runner_activation_fallback(
                "HummingMoeQuantInfo", "w", "fp8", kernel
            )
    else:
        assert (
            activation_fallback.check_runner_activation_fallback(
                "HummingMoeQuantInfo", "w", "fp8", kernel
            )
            is not None
        )


def test_humming_with_input_quant_config_honors_scheme(activation_fallback):
    kernel = _HummingKernel(has_input_quant_config=True)
    assert (
        activation_fallback.check_runner_activation_fallback(
            "HummingMoeQuantInfo", "w", "fp8", kernel
        )
        is None
    )


def test_mxfp4_emulation_honors_scheme(activation_fallback):
    # Emulation kernels execute the requested activation quantization.
    assert (
        activation_fallback.check_runner_activation_fallback(
            "Nvfp4QuantizationEmulationTritonExperts", "w", "nvfp4", _Kernel
        )
        is None
    )


def test_fp8_marlin_fallback_reason(monkeypatch, activation_fallback):
    monkeypatch.delenv("VLLM_STRICT_QUANT_SCHEME", raising=False)
    reason = activation_fallback.check_runner_activation_fallback(
        "MarlinExperts", "fp8_static", "fp8_dynamic", _WeightOnlyKernel
    )
    assert reason is not None
    assert "fp8_dynamic" in reason


# ---------------------------------------------------------------------------
# UNQUANTIZED_ACTIVATION sentinel
# ---------------------------------------------------------------------------


def test_unquantized_activation_sentinel(activation_fallback):
    assert activation_fallback.UNQUANTIZED_ACTIVATION is None
