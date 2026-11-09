"""Per-tensor FP8 quantization edge cases for the MLA KV pack path.

Skips cleanly when torch/triton or a GPU is unavailable so the verification
commands can run in a minimal environment."""

import os
import sys

import pytest

_PY = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "python")
_PY = os.path.abspath(_PY)
if os.path.isdir(_PY) and _PY not in sys.path:
    sys.path.insert(0, _PY)

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm device required", allow_module_level=True)

# The quantizer module imports triton at module scope; guard it so a CPU-only
# or triton-less environment skips instead of erroring during collection.
# Import the module first (importorskip on the module path) so a missing
# optional dependency is a skip, not a collection error.
_quant_mod = pytest.importorskip(
    "sglang.kernels.ops.attention._per_tensor_fp8_quant_triton"
)

fused_per_tensor_fp8_quant = _quant_mod.fused_per_tensor_fp8_quant
reference_per_tensor_fp8_quant = _quant_mod.reference_per_tensor_fp8_quant

FP8_DTYPE = torch.float8_e4m3fn


@pytest.mark.parametrize("numel", [0, 1, 3, 37, 4095, 4096, 4097, 8192])
def test_quant_matches_reference(numel):
    torch.manual_seed(0)
    x = (torch.randn(numel, device="cuda") * 4).to(torch.bfloat16)
    (out,), (descale,) = fused_per_tensor_fp8_quant(x)
    ref_fp8, ref_descale = reference_per_tensor_fp8_quant(x)
    assert out.shape == x.shape and out.is_contiguous()
    torch.testing.assert_close(descale, ref_descale, rtol=1e-6, atol=0)
    ulp = out.view(torch.int8).int() - ref_fp8.view(torch.int8).int()
    assert ulp.numel() == 0 or ulp.abs().max() <= 1


def test_quant_all_zero():
    x = torch.zeros(128, device="cuda", dtype=torch.bfloat16)
    (out,), (descale,) = fused_per_tensor_fp8_quant(x)
    assert torch.all(out.view(torch.int8) == 0)
    assert descale.item() > 0


def test_quant_extreme_values():
    x = torch.tensor([-1e4, 1e4, 0.0], device="cuda", dtype=torch.bfloat16)
    (out,), (descale,) = fused_per_tensor_fp8_quant(x)
    assert descale.item() > 0
    assert torch.isfinite(out.float()).all()


def test_quant_unaligned_length():
    x = torch.full((4097,), 0.5, device="cuda", dtype=torch.bfloat16)
    (out,), (descale,) = fused_per_tensor_fp8_quant(x)
    assert descale.item() > 0
    assert out.numel() == 4097
