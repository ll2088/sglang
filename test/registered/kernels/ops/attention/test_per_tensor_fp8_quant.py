"""Offline tests for per-tensor FP8 quantization and MLA prefill fallback.

These tests are collectable without a GPU or triton: the reference
implementation is pure torch, and GPU-only cases skip per test.
"""

import sys

import pytest

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=30, suite="nightly-amd-kernel-1-gpu", nightly=True)

try:
    import torch
except ImportError:  # pragma: no cover - torch is a hard dep in CI
    torch = None

_HAS_GPU = torch is not None and torch.cuda.is_available()


def _require_gpu() -> None:
    if not _HAS_GPU:
        pytest.skip("per-tensor FP8 quant tests require a CUDA/ROCm device")


def _reference():
    from sglang.kernels.ops.attention.per_tensor_fp8_quant import (
        reference_per_tensor_fp8_quant,
    )

    return reference_per_tensor_fp8_quant


FP8_DTYPE = torch.float8_e4m3fn if torch is not None else None


@pytest.mark.parametrize("num_tokens", [0, 1, 37, 8192])
@pytest.mark.parametrize("num_heads", [1, 12])
def test_reference_quant_matches_manual(num_tokens: int, num_heads: int) -> None:
    _require_gpu()
    reference_per_tensor_fp8_quant = _reference()
    torch.manual_seed(0)
    qk_nope, qk_rope, v_head = 128, 64, 128
    q = torch.randn(num_tokens, num_heads, qk_nope + qk_rope, device="cuda") * 4
    k = torch.randn(num_tokens, num_heads, qk_nope + qk_rope, device="cuda")
    kv = torch.randn(num_tokens, num_heads, qk_nope + v_head, device="cuda")
    v = kv[..., qk_nope:]
    inputs = [x.to(torch.bfloat16) for x in (q, k, v)]
    outputs, descales = reference_per_tensor_fp8_quant(*inputs, fp8_dtype=FP8_DTYPE)
    for x, x_fp8, descale in zip(inputs, outputs, descales):
        amax = x.abs().float().max() if x.numel() else x.new_zeros((), dtype=torch.float)
        ref_descale = amax / 448.0 if amax > 0 else amax.new_tensor(1e-6)
        ref_fp8 = (x.float() / ref_descale).clamp(-448, 448).to(FP8_DTYPE)
        assert x_fp8.shape == x.shape and x_fp8.is_contiguous()
        torch.testing.assert_close(descale, ref_descale.reshape(1), rtol=1e-6, atol=0)
        torch.testing.assert_close(x_fp8.float(), ref_fp8.float(), rtol=0, atol=0)


def test_reference_quant_all_zero_and_extreme() -> None:
    _require_gpu()
    reference_per_tensor_fp8_quant = _reference()
    s, h = 16, 4
    z = torch.zeros(s, h, 128, device="cuda", dtype=torch.bfloat16)
    outs, descales = reference_per_tensor_fp8_quant(z, z, z, fp8_dtype=FP8_DTYPE)
    for o, d in zip(outs, descales):
        assert torch.isfinite(o.float()).all()
        torch.testing.assert_close(d, torch.tensor([1e-6], device="cuda"), rtol=0, atol=0)
    big = torch.full((s, h, 128), 1e4, device="cuda", dtype=torch.bfloat16)
    outs, _ = reference_per_tensor_fp8_quant(big, fp8_dtype=FP8_DTYPE)
    assert outs[0].float().abs().max() <= 448


def test_fused_quant_falls_back_without_triton(monkeypatch) -> None:
    _require_gpu()
    import sglang.kernels.ops.attention.per_tensor_fp8_quant as mod

    def _boom(*args, **kwargs):
        raise ImportError("triton unavailable")

    monkeypatch.setattr(mod, "fused_per_tensor_fp8_quant", mod.reference_per_tensor_fp8_quant)
    x = torch.randn(8, 2, 64, device="cuda", dtype=torch.bfloat16)
    outs, descales = mod.fused_per_tensor_fp8_quant(x, fp8_dtype=FP8_DTYPE)
    assert len(outs) == 1 and len(descales) == 1


def test_mla_prefill_fallback_when_opt_in_disabled() -> None:
    """With the opt-in backend unset, the selector must not pick FlyDSL FP8."""
    from sglang.kernels.ops.attention import registry

    names = {s.backend.value for s in registry.get("attention.fused_per_tensor_fp8_quant")}
    assert "triton" in names
    # No FlyDSL FP8 prefill backend is registered by default.
    assert not any("flydsl_fp8" in s.op for s in registry.all_specs())


def test_attention_module_import_smoke() -> None:
    import sglang.kernels.ops.attention as m  # noqa: F401

    assert hasattr(m, "registry")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
