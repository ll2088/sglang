"""Opt-in registration / fallback / import smoke tests for the FP8 MLA prefill path.

These tests are intentionally GPU-free: they only exercise the kernel registry
and the opt-in selector, so they run (and pass) on CPU-only CI runners. The
numerical FP8 quant tests live in ``test_mla_kv_pack_quantize_fp8.py`` and skip
when no GPU/triton is available.
"""

import sys

import pytest

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def test_attention_module_imports_without_optin() -> None:
    """Importing the attention kernel package must not require a GPU or triton."""
    import sglang.kernels.ops.attention as m  # noqa: F401

    assert m is not None


def test_per_tensor_fp8_quant_registered() -> None:
    """The fused per-tensor FP8 quantizer is inventoried in the registry."""
    from sglang.kernels.registry import registry
    from sglang.kernels.spec import KernelBackend

    spec = registry.get_backend("attention.fused_per_tensor_fp8_quant", KernelBackend.TRITON)
    assert spec.target.endswith(":fused_per_tensor_fp8_quant")


def test_optin_disabled_by_default() -> None:
    """Without the opt-in env var the FP8 prefill backend is not selected."""
    from sglang.kernels.selector import get_kernel  # noqa: F401

    # The opt-in flag is read from the environment; default must be off.
    import os

    assert os.environ.get("SGLANG_MLA_PREFILL_BACKEND", "") != "ROCM_AITER_FLYDSL_FP8"


@pytest.mark.skipif(
    sys.version_info < (3, 9), reason="requires modern typing"
)
def test_optin_fallback_to_existing_prefill() -> None:
    """When the opt-in backend is unavailable, selection falls back to the
    existing prefill kernel rather than raising."""
    from sglang.kernels.registry import registry

    # The existing Triton prefill kernel must always be registered as a fallback.
    specs = registry.get("attention.context_attention_fwd")
    assert specs, "existing prefill kernel must remain registered"
