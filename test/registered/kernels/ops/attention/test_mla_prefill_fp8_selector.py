"""Opt-in FlyDSL FP8 MLA prefill: registration, fallback and import smoke.

These checks must run without a GPU, torch kernels or aiter installed: the
selector is pure metadata, so the default (opt-in off) path must resolve to the
parent prefill backend and importing the attention package must not pull in
optional dependencies.
"""

import os
import sys

import pytest

_PY = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "python")
_PY = os.path.abspath(_PY)
if os.path.isdir(_PY) and _PY not in sys.path:
    sys.path.insert(0, _PY)


def test_attention_package_imports_without_optional_deps():
    """Importing the attention group must not require torch/triton/aiter."""
    import sglang.kernels.ops.attention as attention

    assert attention is not None


def test_registry_lists_flydsl_fp8_prefill_selector():
    from sglang.kernels.registry import registry

    ops = registry.ops()
    assert "attention.select_mla_prefill_backend" in ops
    assert "attention.resolve_mla_prefill_kernel" in ops


def test_selector_defaults_to_parent_backend():
    """With opt-in off, the selector must not pick the FlyDSL FP8 backend."""
    selector = pytest.importorskip(
        "sglang.kernels.ops.attention.mla_prefill_fp8_selector"
    )
    select = getattr(selector, "select_mla_prefill_backend", None)
    if select is None:
        pytest.skip("selector entry point not present in this build")
    # No explicit opt-in: the selector must return a non-FlyDSL backend name.
    name = select(enable_flydsl_fp8=False)
    assert "FLYDSL_FP8" not in str(name).upper()


def test_selector_opt_in_resolves_flydsl_fp8():
    selector = pytest.importorskip(
        "sglang.kernels.ops.attention.mla_prefill_fp8_selector"
    )
    select = getattr(selector, "select_mla_prefill_backend", None)
    if select is None:
        pytest.skip("selector entry point not present in this build")
    name = select(enable_flydsl_fp8=True)
    assert "FLYDSL_FP8" in str(name).upper()
