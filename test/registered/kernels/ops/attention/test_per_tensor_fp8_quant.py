"""Per-tensor FP8 quantization + opt-in MLA prefill backend registration.

Import-safe on CPU-only / triton-less / torch-less environments: the module
skips instead of raising at import time, so the acceptance import smoke test
and the registry tests can run without a GPU.
"""

import sys

import pytest

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=60, suite="nightly-amd-kernel-1-gpu", nightly=True)

try:
    import torch
except ImportError:  # pragma: no cover - torch is a hard dep in CI
    torch = None

_HAS_GPU = torch is not None and torch.cuda.is_available()
_HAS_TRITON = False
if _HAS_GPU:
    try:
        import triton  # noqa: F401

        _HAS_TRITON = True
    except ImportError:
        _HAS_TRITON = False

DEVICE = "cuda"
FP8_DTYPE = None
if torch is not None:
    FP8_DTYPE = torch.float8_e4m3fn


def _require_gpu() -> None:
    if not _HAS_GPU:
        pytest.skip("per-tensor FP8 quant requires a CUDA/ROCm device")
    if not _HAS_TRITON:
        pytest.skip("per-tensor FP8 quant requires triton")


def _quant():
    from sglang.kernels.ops.attention.per_tensor_fp8_quant import (
        fused_per_tensor_fp8_quant,
    )

    return fused_per_tensor_fp8_quant


def _reference(x: "torch.Tensor"):
    amax = x.abs().float().max() if x.numel() else x.new_zeros((), dtype=torch.float)
    descale = (
        amax / torch.finfo(FP8_DTYPE).max if amax > 0 else amax.new_tensor(1e-6)
    )
    x_fp8 = (x.float() / descale).clamp(-448, 448).to(FP8_DTYPE)
    return x_fp8, descale.reshape(1)


@pytest.mark.parametrize("num_tokens", [0, 1, 37, 8192])
def test_fused_per_tensor_fp8_quant_matches_reference(num_tokens: int) -> None:
    _require_gpu()
    fused_per_tensor_fp8_quant = _quant()
    torch.manual_seed(0)
    q = torch.randn(num_tokens, 12, 192, device=DEVICE) * 4
    k = torch.randn(num_tokens, 12, 192, device=DEVICE)
    kv = torch.randn(num_tokens, 12, 256, device=DEVICE)
    v = kv[..., 128:]
    inputs = [x.to(torch.bfloat16) for x in (q, k, v)]
    outputs, descales = fused_per_tensor_fp8_quant(*inputs)
    for x, x_fp8, descale in zip(inputs, outputs, descales):
        ref_fp8, ref_descale = _reference(x)
        assert x_fp8.shape == x.shape and x_fp8.is_contiguous()
        torch.testing.assert_close(descale, ref_descale, rtol=1e-6, atol=0)
        ulp_diff = x_fp8.view(torch.int8).int() - ref_fp8.view(torch.int8).int()
        assert ulp_diff.numel() == 0 or ulp_diff.abs().max() <= 1


def test_fused_per_tensor_fp8_quant_all_zero_and_extreme() -> None:
    _require_gpu()
    fused_per_tensor_fp8_quant = _quant()
    x = torch.zeros(16, 4, 128, device=DEVICE, dtype=torch.bfloat16)
    outputs, descales = fused_per_tensor_fp8_quant(x, x, x)
    for out, descale in zip(outputs, descales):
        assert torch.isfinite(out.float()).all()
        assert (descale > 0).all()
    x.fill_(1e4)
    outputs, _ = fused_per_tensor_fp8_quant(x, x, x)
    for out in outputs:
        assert out.float().abs().max() <= 448


def test_opt_in_backend_registered_and_default_off() -> None:
    """Registry resolves the opt-in backend only when explicitly requested."""
    from sglang.kernels.registry import registry
    from sglang.kernels.spec import KernelBackend

    op = "attention.fused_per_tensor_fp8_quant"
    assert registry.has(op)
    spec = registry.get_backend(op, KernelBackend.TRITON)
    assert spec.target.endswith(":fused_per_tensor_fp8_quant")
    # Default (no opt-in) must not resolve a FlyDSL FP8 prefill backend.
    names = {s.backend for s in registry.get(op)}
    assert KernelBackend.FLYDSL not in names


def test_import_smoke() -> None:
    import sglang.kernels.ops.attention as m  # noqa: F401

    assert m is not None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
