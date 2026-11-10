"""Per-tensor FP8 quantization: import safety, numerics and opt-in gating.

Collection must succeed on CPU-only hosts and without triton installed, so the
module-level guards below only skip the GPU cases instead of failing setup.
"""

import pytest
import torch

# Collection must not import triton: the module is optional on CPU-only hosts.
q = pytest.importorskip(
    "sglang.kernels.ops.attention._per_tensor_fp8_quant_triton",
    reason="per-tensor FP8 quant module needs triton",
)

_HAS_CUDA = torch.cuda.is_available()


def test_import_ok_without_opt_in():
    """Importing the attention package must not require aiter/flydsl/triton."""
    import sglang.kernels.ops.attention  # noqa: F401

    assert hasattr(q, "fused_per_tensor_fp8_quant")
    assert hasattr(q, "reference_per_tensor_fp8_quant")


def test_attention_package_imports_without_triton():
    """The registry package import must not pull in triton or a GPU."""
    import sglang.kernels.ops.attention as m

    assert m is not None


def test_reference_quant_all_zero_and_extreme():
    for x in (
        torch.zeros(7, dtype=torch.bfloat16),
        torch.tensor([1e4, -1e4, 0.0], dtype=torch.bfloat16),
        torch.empty(0, dtype=torch.bfloat16),
    ):
        fp8, descale = q.reference_per_tensor_fp8_quant(x)
        assert fp8.shape == x.shape
        assert descale.numel() == 1 and descale.item() > 0


@pytest.mark.skipif(not _HAS_CUDA, reason="needs a CUDA/ROCm device")
@pytest.mark.skipif(not q._HAS_TRITON, reason="needs triton")
@pytest.mark.parametrize("numel", [0, 1, 37, 4096, 4097])
def test_fused_matches_reference(numel):
    x = (torch.randn(numel, device="cuda") * 4).to(torch.bfloat16)
    (out,), (descale,) = q.fused_per_tensor_fp8_quant(x)
    ref, ref_descale = q.reference_per_tensor_fp8_quant(x)
    torch.testing.assert_close(descale, ref_descale, rtol=1e-6, atol=0)
    diff = out.view(torch.int8).int() - ref.view(torch.int8).int()
    assert diff.numel() == 0 or diff.abs().max() <= 1


@pytest.mark.skipif(not _HAS_CUDA, reason="needs a CUDA/ROCm device")
@pytest.mark.skipif(not q._HAS_TRITON, reason="needs triton")
def test_fused_requires_cuda_tensor():
    with pytest.raises(RuntimeError):
        q.fused_per_tensor_fp8_quant(torch.zeros(4, dtype=torch.bfloat16))
