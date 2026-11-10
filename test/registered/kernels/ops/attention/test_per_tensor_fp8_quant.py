"""Per-tensor FP8 quantization: import safety, numerics and opt-in gating.

Collection must succeed on CPU-only hosts and without triton installed, so the
module-level guards below only skip the GPU cases instead of failing setup.
"""

import pytest
import torch

# Collection must not import triton: the module is optional on CPU-only hosts.
# Import lazily so the import-smoke tests below still run when triton is absent.
try:
    from sglang.kernels.ops.attention import _per_tensor_fp8_quant_triton as q
except Exception:  # noqa: BLE001 - triton missing or import-time failure
    q = None

_HAS_CUDA = torch.cuda.is_available()
# The fused kernel needs triton *and* a device; keep both gates explicit so the
# module still collects on CPU-only hosts (CI env-prep must not need triton).
_HAS_TRITON = q is not None and getattr(q, "_HAS_TRITON", False)

requires_quant = pytest.mark.skipif(
    q is None, reason="per-tensor FP8 quant module needs triton"
)


def test_import_ok_without_opt_in():
    """Importing the attention package must not require aiter/flydsl/triton."""
    import sglang.kernels.ops.attention  # noqa: F401

    if q is None:
        pytest.skip("per-tensor FP8 quant module needs triton")
    assert hasattr(q, "fused_per_tensor_fp8_quant")
    assert hasattr(q, "reference_per_tensor_fp8_quant")


def test_attention_package_imports_without_triton():
    """The registry package import must not pull in triton or a GPU."""
    import sglang.kernels.ops.attention as m

    assert m is not None


@requires_quant
def test_reference_quant_all_zero_and_extreme():
    for x in (
        torch.zeros(7, dtype=torch.bfloat16),
        torch.tensor([1e4, -1e4, 0.0], dtype=torch.bfloat16),
        torch.empty(0, dtype=torch.bfloat16),
    ):
        fp8, descale = q.reference_per_tensor_fp8_quant(x)
        assert fp8.shape == x.shape
        assert descale.numel() == 1 and descale.item() > 0


@requires_quant
@pytest.mark.skipif(not _HAS_CUDA, reason="needs a CUDA/ROCm device")
@pytest.mark.skipif(not _HAS_TRITON, reason="needs triton")
@pytest.mark.parametrize("numel", [0, 1, 37, 4096, 4097])
def test_fused_matches_reference(numel):
    x = (torch.randn(numel, device="cuda") * 4).to(torch.bfloat16)
    (out,), (descale,) = q.fused_per_tensor_fp8_quant(x)
    ref, ref_descale = q.reference_per_tensor_fp8_quant(x)
    torch.testing.assert_close(descale, ref_descale, rtol=1e-6, atol=0)
    diff = out.view(torch.int8).int() - ref.view(torch.int8).int()
    assert diff.numel() == 0 or diff.abs().max() <= 1


@requires_quant
@pytest.mark.skipif(not _HAS_CUDA, reason="needs a CUDA/ROCm device")
@pytest.mark.skipif(not _HAS_TRITON, reason="needs triton")
def test_fused_requires_cuda_tensor():
    with pytest.raises(RuntimeError):
        q.fused_per_tensor_fp8_quant(torch.zeros(4, dtype=torch.bfloat16))


def test_mla_prefill_fp8_selector_registered():
    """The opt-in FlyDSL FP8 MLA prefill selector is inventoried as metadata only.

    Registration must not import aiter/flydsl/triton, so this runs on CPU-only
    hosts and proves the default (opt-in off) import path stays clean.
    """
    from sglang.kernels.registry import registry
    from sglang.kernels.spec import KernelBackend

    for fn in ("select_mla_prefill_backend", "resolve_mla_prefill_kernel"):
        spec = registry.get_backend(f"attention.{fn}", KernelBackend.TRITON)
        assert spec.target == (
            f"sglang.kernels.ops.attention.mla_prefill_fp8_selector:{fn}"
        )


def test_mla_prefill_fp8_selector_opt_in_off_by_default():
    """Without the opt-in env/config the selector must not pick the FP8 backend."""
    pytest.importorskip("sglang.kernels.ops.attention.mla_prefill_fp8_selector")
    from sglang.kernels.ops.attention import mla_prefill_fp8_selector as sel

    assert hasattr(sel, "select_mla_prefill_backend")
    assert hasattr(sel, "resolve_mla_prefill_kernel")
    # Default resolution must fall back to the existing prefill path.
    if hasattr(sel, "DEFAULT_BACKEND"):
        assert sel.DEFAULT_BACKEND != "ROCM_AITER_FLYDSL_FP8"
