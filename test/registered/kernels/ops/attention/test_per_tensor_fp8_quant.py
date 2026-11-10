"""Tests for the fused per-tensor FP8 quantizer and the opt-in FlyDSL FP8
MLA prefill selector registration.

These tests must be safe to collect on CPU-only hosts without torch/triton or
the optional aiter/flydsl packages: the kernel tests skip, and the registry
checks only read metadata (they never import the target modules).
"""

from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")

from sglang.kernels.ops.attention import _per_tensor_fp8_quant_triton as q  # noqa: E402
from sglang.kernels.registry import registry  # noqa: E402
from sglang.kernels.spec import KernelBackend  # noqa: E402

needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="per-tensor FP8 quant needs a GPU"
)


def _reference(x: torch.Tensor):
    amax = x.abs().float().max() if x.numel() else x.new_zeros((), dtype=torch.float)
    descale = amax / q.FP8_MAX if amax > 0 else amax.new_tensor(q.MIN_DESCALE)
    x_fp8 = (x.float() / descale).clamp(-q.FP8_MAX, q.FP8_MAX).to(torch.float8_e4m3fn)
    return x_fp8, descale.reshape(1)


@needs_gpu
@pytest.mark.parametrize("numel", [0, 1, 37, 4096, 4097, 8192])
def test_fused_per_tensor_fp8_quant_matches_reference(numel: int) -> None:
    torch.manual_seed(0)
    x = (torch.randn(numel, device="cuda") * 4).to(torch.bfloat16)
    (out,), (descale,) = q.fused_per_tensor_fp8_quant(x)
    ref_fp8, ref_descale = _reference(x)
    assert out.shape == x.shape and out.is_contiguous()
    torch.testing.assert_close(descale, ref_descale, rtol=1e-6, atol=0)
    ulp = out.view(torch.int8).int() - ref_fp8.view(torch.int8).int()
    assert ulp.numel() == 0 or ulp.abs().max() <= 1


@needs_gpu
@pytest.mark.parametrize("value", [0.0, 1e-30, 1e30, -1e30])
def test_fused_per_tensor_fp8_quant_edge_values(value: float) -> None:
    x = torch.full((64,), value, device="cuda", dtype=torch.bfloat16)
    (out,), (descale,) = q.fused_per_tensor_fp8_quant(x)
    assert torch.isfinite(descale).all() and (descale > 0).all()
    assert out.dtype == torch.float8_e4m3fn


@needs_gpu
def test_fused_per_tensor_fp8_quant_multiple_inputs() -> None:
    a = torch.randn(100, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(7, device="cuda", dtype=torch.bfloat16)
    outs, descales = q.fused_per_tensor_fp8_quant(a, b)
    assert len(outs) == len(descales) == 2
    assert outs[0].shape == a.shape and outs[1].shape == b.shape


def test_fused_per_tensor_fp8_quant_rejects_cpu() -> None:
    with pytest.raises(RuntimeError, match="CUDA/ROCm"):
        q.fused_per_tensor_fp8_quant(torch.zeros(4, dtype=torch.bfloat16))


def test_flydsl_fp8_selector_registered_once() -> None:
    """The opt-in selector is inventoried as metadata only, exactly once."""
    for fn in ("select_mla_prefill_backend", "resolve_mla_prefill_kernel"):
        specs = [
            s
            for s in registry.get(f"attention.{fn}")
            if s.backend == KernelBackend.TRITON
        ]
        assert len(specs) == 1, (fn, specs)
        assert specs[0].target == (
            f"sglang.kernels.ops.attention.mla_prefill_fp8_selector:{fn}"
        )


def test_flydsl_fp8_selector_not_imported_eagerly() -> None:
    """Registering the selector must not import it (or aiter/flydsl)."""
    import sys

    assert "sglang.kernels.ops.attention.mla_prefill_fp8_selector" not in sys.modules
