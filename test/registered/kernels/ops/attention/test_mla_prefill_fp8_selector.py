"""Opt-in FlyDSL FP8 MLA prefill selector: default-off and explicit-on.

Runs on CPU-only hosts: the selector module is imported lazily and its
availability probe is monkeypatched, so no aiter/flydsl/GPU is required.
"""

import pytest


@pytest.fixture()
def selector():
    return pytest.importorskip(
        "sglang.kernels.ops.attention.mla_prefill_fp8_selector",
        reason="selector module not present in this tree",
    )


def test_default_config_does_not_select_flydsl_fp8(selector, monkeypatch):
    """Without opt-in the selector must not resolve the FP8 backend."""
    monkeypatch.setattr(selector, "_flydsl_fp8_available", lambda *a, **k: True)
    backend = selector.select_mla_prefill_backend(enabled=False)
    assert backend != "ROCM_AITER_FLYDSL_FP8"


def test_explicit_opt_in_selects_flydsl_fp8(selector, monkeypatch):
    """Explicit opt-in plus availability resolves the FP8 backend."""
    monkeypatch.setattr(selector, "_flydsl_fp8_available", lambda *a, **k: True)
    backend = selector.select_mla_prefill_backend(enabled=True)
    assert backend == "ROCM_AITER_FLYDSL_FP8"


def test_opt_in_without_availability_falls_back(selector, monkeypatch):
    """Opt-in on an unsupported device falls back to the parent path."""
    monkeypatch.setattr(selector, "_flydsl_fp8_available", lambda *a, **k: False)
    backend = selector.select_mla_prefill_backend(enabled=True)
    assert backend != "ROCM_AITER_FLYDSL_FP8"
