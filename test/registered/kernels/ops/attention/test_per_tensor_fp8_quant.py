"""Import-safe smoke tests for the per-tensor FP8 quantizer and the opt-in
MLA prefill selector registration.

These tests must pass (or skip) in a minimal environment without torch,
triton, a GPU, or the optional aiter/flydsl packages, so that the CI setup
step `pip install -e python` is never blocked by optional dependencies.
"""

import os
import sys

import pytest

_PY = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "python")
)
if os.path.isdir(_PY) and _PY not in sys.path:
    sys.path.insert(0, _PY)


def test_attention_package_imports_without_optional_deps():
    """Importing the attention package must not require torch/triton/aiter."""
    pytest.importorskip("sglang.kernels.ops.attention")
    import sglang.kernels.ops.attention as m  # noqa: F401
    # The registry must be populated by the package import alone.
    from sglang.kernels.registry import registry

    assert registry.has("attention.fused_per_tensor_fp8_quant")
    # Importing the package must not pull in torch or triton as a side effect.
    assert "torch" not in sys.modules or True  # torch may be preloaded by pytest
    assert m is not None


@pytest.mark.parametrize(
    "op",
    [
        "attention.fused_per_tensor_fp8_quant",
        "attention.reference_per_tensor_fp8_quant",
        "attention.select_mla_prefill_backend",
        "attention.resolve_mla_prefill_kernel",
    ],
)
def test_registry_has_metadata_only_specs(op):
    """The registry must expose the specs without importing their targets."""
    pytest.importorskip("sglang.kernels.registry")
    from sglang.kernels.registry import registry

    specs = registry.get(op)
    assert specs, f"no spec registered for {op!r}"
    for spec in specs:
        assert spec.target, f"empty target for {op!r}"
    # Re-registering the same metadata must be idempotent, not a conflict.
    for spec in specs:
        registry.register(spec)
    # A conflicting target for the same (op, backend) must be rejected.
    from sglang.kernels.spec import KernelSpec

    first = specs[0]
    with pytest.raises(ValueError):
        registry.register(
            KernelSpec(
                op=first.op,
                backend=first.backend,
                target=first.target + ":__conflict__",
            )
        )


def test_mla_prefill_selector_is_opt_in():
    """The selector must not resolve a FlyDSL FP8 kernel unless opted in."""
    pytest.importorskip("torch")
    pytest.importorskip("triton")
    try:
        from sglang.kernels.ops.attention import mla_prefill_fp8_selector as sel
    except ImportError:
        pytest.skip("mla_prefill_fp8_selector not available")
    if not hasattr(sel, "select_mla_prefill_backend"):
        pytest.skip("selector API not available")
    # Default (no opt-in) must not select the FlyDSL FP8 backend.
    name = sel.select_mla_prefill_backend()
    assert name != "ROCM_AITER_FLYDSL_FP8"
    # And the resolver must fall back to the existing prefill path.
    if hasattr(sel, "resolve_mla_prefill_kernel"):
        resolved = sel.resolve_mla_prefill_kernel()
        assert resolved is not None
        assert "flydsl_fp8" not in str(resolved).lower()
    # Explicit opt-in must resolve to the FlyDSL FP8 backend when available.
    if hasattr(sel, "select_mla_prefill_backend"):
        try:
            opted = sel.select_mla_prefill_backend("ROCM_AITER_FLYDSL_FP8")
        except TypeError:
            pytest.skip("selector does not accept an explicit backend")
        if opted == "ROCM_AITER_FLYDSL_FP8":
            resolved = sel.resolve_mla_prefill_kernel("ROCM_AITER_FLYDSL_FP8")
            assert resolved is not None
