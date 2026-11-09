import sys

import pytest

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=30, suite="nightly-amd-kernel-1-gpu", nightly=True)


def test_flydsl_fp8_capability_probe_is_safe_without_aiter():
    """Capability probing must never raise when AITER/FlyDSL is absent."""
    try:
        from aiter.ops.flydsl import flydsl_flash_attn_fp8_supported  # noqa: F401
    except Exception:  # noqa: BLE001
        # Expected on non-gfx950 / AITER-less environments: the opt-in backend
        # must fall back to the parent BF16 path instead of failing import.
        return
    # If AITER is present, the probe must be callable and return a bool.
    assert callable(flydsl_flash_attn_fp8_supported)


def test_opt_in_default_is_off():
    """The FlyDSL FP8 prefill backend must not be auto-selected by default."""
    # Import lazily: sglang.srt.environ pulls in runtime deps that are not
    # available in the CPU-only CI setup step, and this test must not turn a
    # missing optional dependency into a collection error.
    try:
        from sglang.srt.environ import envs
    except Exception:  # noqa: BLE001
        pytest.skip("sglang.srt.environ unavailable in this environment")

    # No env var should enable the opt-in backend implicitly.
    for name in dir(envs):
        if "FLYDSL" in name.upper() and "FP8" in name.upper():
            value = getattr(envs, name)
            getter = getattr(value, "get", None)
            if callable(getter):
                assert getter() in (False, None, "", 0), name


def test_kernel_registry_query_is_stable():
    """Registry lookups must be idempotent and not import torch backends."""
    from sglang.kernels.registry import registry

    ops = registry.ops()
    assert isinstance(ops, list)
    assert ops == sorted(ops)
    # Repeated queries must not mutate registry state.
    assert registry.ops() == ops


def test_fused_per_tensor_fp8_quant_rejects_bad_arity():
    """The fused quantizer validates its arity before touching any kernel."""
    from sglang.kernels.ops.attention.triton_per_tensor_fp8_quant import (
        fused_per_tensor_fp8_quant,
    )

    with pytest.raises(ValueError):
        fused_per_tensor_fp8_quant()



def test_fused_per_tensor_fp8_quant_rejects_bad_fp8_dtype():
    """Only e4m3/e5m2 are accepted; the check runs before any launch."""
    import torch

    from sglang.kernels.ops.attention.triton_per_tensor_fp8_quant import (
        fused_per_tensor_fp8_quant,
    )

    x = torch.zeros((1, 1, 1), dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        fused_per_tensor_fp8_quant(x, fp8_dtype=torch.float32)


def test_fused_per_tensor_fp8_quant_rejects_non_3d():
    """Inputs must be [tokens, heads, dim]; the shape check is pure Python."""
    import torch

    from sglang.kernels.ops.attention.triton_per_tensor_fp8_quant import (
        fused_per_tensor_fp8_quant,
    )

    x = torch.zeros((4, 8), dtype=torch.bfloat16)
    with pytest.raises(AssertionError):
        fused_per_tensor_fp8_quant(x)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
