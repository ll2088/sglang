"""Tests for fused per-tensor FP8 quantization and the opt-in MLA prefill
backend selector. Skips cleanly when torch/triton or a GPU is unavailable."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm device required", allow_module_level=True)

from sglang.kernels.ops.attention._per_tensor_fp8_quant_triton import (  # noqa: E402
    _BLOCK,
    _PARTS,
    fused_per_tensor_fp8_quant,
    reference_per_tensor_fp8_quant,
)
from sglang.kernels.ops.attention.mla_prefill_fp8_selector import (  # noqa: E402
    BACKEND_NAME,
    resolve_mla_prefill_kernel,
    select_mla_prefill_backend,
)

FP8_DTYPE = torch.float8_e4m3fn
_STRIPE = _PARTS * _BLOCK


@pytest.mark.parametrize("numel", [0, 1, 37, 8192])
def test_fused_per_tensor_fp8_quant_matches_reference(numel):
    torch.manual_seed(0)
    x = (torch.randn(numel, device="cuda") * 4).to(torch.bfloat16)
    (out,), (descale,) = fused_per_tensor_fp8_quant(x)
    ref_fp8, ref_descale = reference_per_tensor_fp8_quant(x)
    assert out.shape == x.shape and out.is_contiguous()
    torch.testing.assert_close(descale, ref_descale, rtol=1e-6, atol=0)
    ulp = out.view(torch.int8).int() - ref_fp8.view(torch.int8).int()
    assert ulp.numel() == 0 or ulp.abs().max() <= 1


@pytest.mark.parametrize("numel", [1, _BLOCK, _BLOCK + 1, _STRIPE - 1, _STRIPE, _STRIPE + 1])
def test_fused_per_tensor_fp8_quant_covers_every_element(numel):
    x = torch.full((numel,), 0.5, device="cuda", dtype=torch.bfloat16)
    (out,), (descale,) = fused_per_tensor_fp8_quant(x)
    expected = 0.5 / (0.5 / 448.0)
    assert torch.allclose(out.float(), torch.full_like(out.float(), expected), atol=1.0)
    assert descale.item() > 0


def test_fused_per_tensor_fp8_quant_all_zero():
    x = torch.zeros(128, device="cuda", dtype=torch.bfloat16)
    (out,), (descale,) = fused_per_tensor_fp8_quant(x)
    assert torch.all(out.view(torch.int8) == 0)
    assert descale.item() > 0


def test_selector_default_off():
    assert select_mla_prefill_backend() is None
    assert select_mla_prefill_backend("ROCM_AITER_FA") is None
    assert resolve_mla_prefill_kernel() is None


def test_selector_opt_in_requires_support():
    # On non-gfx950 or without aiter, opt-in must fall back to the existing path.
    result = select_mla_prefill_backend(BACKEND_NAME)
    if result is None:
        assert resolve_mla_prefill_kernel() is None
    else:
        assert result == BACKEND_NAME


def test_import_smoke():
    import sglang.kernels.ops.attention as m

    assert hasattr(m, "register_kernel")
