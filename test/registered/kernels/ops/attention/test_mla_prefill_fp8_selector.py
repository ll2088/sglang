"""Opt-in MLA prefill backend registration and fallback tests.

These tests do not require a GPU: they only exercise the selector/registry
contract, so they run in the minimal verification environment."""

import os
import sys

import pytest

_PY = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "python")
_PY = os.path.abspath(_PY)
if os.path.isdir(_PY) and _PY not in sys.path:
    sys.path.insert(0, _PY)

pytest.importorskip("torch")

from sglang.kernels.ops.attention.mla_prefill_fp8_selector import (  # noqa: E402
    BACKEND_NAME,
    resolve_mla_prefill_kernel,
    select_mla_prefill_backend,
)


def test_default_off():
    assert select_mla_prefill_backend() is None
    assert resolve_mla_prefill_kernel() is None


def test_existing_backend_not_overridden():
    assert select_mla_prefill_backend("ROCM_AITER_FA") is None
    assert resolve_mla_prefill_kernel("ROCM_AITER_FA") is None


def test_unknown_backend_falls_back():
    assert select_mla_prefill_backend("NOT_A_BACKEND") is None
    assert resolve_mla_prefill_kernel("NOT_A_BACKEND") is None


def test_opt_in_requires_support():
    # Without gfx950 + aiter FlyDSL FP8, opt-in must fall back to the existing path.
    result = select_mla_prefill_backend(BACKEND_NAME)
    if result is None:
        assert resolve_mla_prefill_kernel() is None
    else:
        assert result == BACKEND_NAME
        assert resolve_mla_prefill_kernel(BACKEND_NAME) is not None


def test_registry_import_smoke():
    import sglang.kernels.ops.attention as m

    assert hasattr(m, "register_kernel")
