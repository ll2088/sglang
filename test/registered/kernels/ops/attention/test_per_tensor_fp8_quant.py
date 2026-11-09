"""Per-tensor FP8 quantizer and opt-in MLA prefill backend registration.

Runs on CPU (torch reference path) so the default CI environment can execute
it; the Triton path is exercised only when a CUDA/ROCm device is present.
"""

import sys

import pytest

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=30, suite="nightly-amd-kernel-1-gpu", nightly=True)

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

_HAS_GPU = torch is not None and torch.cuda.is_available()
FP8_DTYPE = torch.float8_e4m3fn if torch is not None else None


def _quant():
    from sglang.kernels.ops.attention.per_tensor_fp8_quant import (
        fused_per_tensor_fp8_quant,
        reference_per_tensor_fp8_quant,
    )

    return fused_per_tensor_fp8_quant, reference_per_tensor_fp8_quant


def _require_torch():
    if torch is None:
        pytest.skip("per-tensor fp8 quant requires torch")


@pytest.mark.parametrize("num_tokens", [0, 1, 37, 8192])
@pytest.mark.parametrize("num_heads", [1, 12])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_per_tensor_fp8_quant_matches_reference(num_tokens, num_heads, device):
    _require_torch()
    if device == "cuda" and not _HAS_GPU:
        pytest.skip("cuda device unavailable")
    fused, reference = _quant()
    torch.manual_seed(0)
    q = torch.randn(num_tokens, num_heads, 192, device=device) * 4
    k = torch.randn(num_tokens, num_heads, 192, device=device)
    kv = torch.randn(num_tokens, num_heads, 256, device=device)
    v = kv[..., 128:]
    inputs = [x.to(torch.bfloat16) for x in (q, k, v)]
    out, desc = fused(*inputs)
    ref_out, ref_desc = reference(*inputs)
    for got, want, d, rd in zip(out, ref_out, desc, ref_desc):
        assert got.shape == want.shape and got.is_contiguous()
        assert d.shape == (1,) and torch.isfinite(d).all() and d > 0
        torch.testing.assert_close(d, rd, rtol=1e-6, atol=0)
        ulp = got.view(torch.int8).int() - want.view(torch.int8).int()
        assert ulp.numel() == 0 or ulp.abs().max() <= 1


def test_per_tensor_fp8_quant_all_zero_and_extreme():
    _require_torch()
    fused, _ = _quant()
    x = torch.zeros(16, 4, 128, dtype=torch.bfloat16)
    out, desc = fused(x)
    assert torch.isfinite(out[0].float()).all()
    assert desc[0] > 0
    x.fill_(1e4)
    out, _ = fused(x)
    assert out[0].float().abs().max() <= 448


def test_per_tensor_fp8_quant_rejects_too_many_inputs():
    _require_torch()
    fused, _ = _quant()
    x = torch.zeros(1, 1, 1, dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        fused(x, x, x, x)


def test_attention_module_imports_without_opt_in():
    import sglang.kernels.ops.attention as m  # noqa: F401

    from sglang.kernels.registry import registry

    assert registry.has("attention.fused_per_tensor_fp8_quant")
    assert registry.has("attention.reference_per_tensor_fp8_quant")


def test_mla_kv_pack_quantize_fp8_registered():
    from sglang.kernels.registry import registry
    from sglang.kernels.spec import KernelBackend

    spec = registry.get_backend("attention.mla_kv_pack_quantize_fp8", KernelBackend.TRITON)
    assert spec.target.endswith(":mla_kv_pack_quantize_fp8")


def test_opt_in_backend_not_selected_by_default():
    """The FlyDSL FP8 prefill backend is opt-in: absent from the default set."""
    from sglang.kernels.registry import registry

    names = {s.backend.value for s in registry.get("attention.fused_per_tensor_fp8_quant")}
    assert "flydsl" not in names


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
