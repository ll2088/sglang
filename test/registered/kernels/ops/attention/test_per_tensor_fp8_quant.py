import sys

import pytest
import torch

from sglang.kernels.jit.utils import get_ci_test_range
from sglang.kernels.ops.attention.per_tensor_fp8_quant import (
    _BLOCK,
    _PARTS,
    fused_per_tensor_fp8_quant,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=60, suite="nightly-amd-kernel-1-gpu", nightly=True)

# The kernel needs a CUDA/ROCm device; skip collection on CPU-only CI so the
# module still imports (import-smoke dimension) instead of erroring at import.
if not torch.cuda.is_available():
    pytest.skip("per-tensor FP8 quant requires a CUDA/ROCm device", allow_module_level=True)

DEVICE = "cuda"
FP8_DTYPE = torch.float8_e4m3fn


def _reference_quant(x: torch.Tensor):
    amax = x.abs().float().max() if x.numel() else x.new_zeros((), dtype=torch.float)
    descale = amax / torch.finfo(FP8_DTYPE).max if amax > 0 else amax.new_tensor(1e-6)
    x_fp8 = (x.float() / descale).clamp(-448, 448).to(FP8_DTYPE)
    return x_fp8, descale.reshape(1)


@pytest.mark.parametrize("num_tokens", get_ci_test_range([0, 1, 37, 8192], [0, 1, 8192]))
def test_matches_reference(num_tokens: int) -> None:
    torch.manual_seed(0)
    q = torch.randn(num_tokens, 12, 192, device=DEVICE) * 4
    k = torch.randn(num_tokens, 12, 192, device=DEVICE)
    kv = torch.randn(num_tokens, 12, 256, device=DEVICE)
    v = kv[..., 128:]
    inputs = [x.to(torch.bfloat16) for x in (q, k, v)]
    outputs, descales = fused_per_tensor_fp8_quant(*inputs)
    for x, x_fp8, descale in zip(inputs, outputs, descales):
        ref_fp8, ref_descale = _reference_quant(x)
        assert x_fp8.shape == x.shape and x_fp8.is_contiguous()
        torch.testing.assert_close(descale, ref_descale, rtol=1e-6, atol=0)
        ulp_diff = x_fp8.view(torch.int8).int() - ref_fp8.view(torch.int8).int()
        assert ulp_diff.numel() == 0 or ulp_diff.abs().max() <= 1


_STRIPE = _PARTS * _BLOCK


@pytest.mark.parametrize(
    "numel",
    get_ci_test_range(
        [1, _BLOCK, _BLOCK + 1, _STRIPE - 1, _STRIPE, _STRIPE + 1, 3 * _STRIPE + 7],
        [1, _BLOCK + 1, _STRIPE + 1],
    ),
)
def test_covers_every_element(numel: int) -> None:
    x = torch.full((numel, 1, 1), 0.5, device=DEVICE, dtype=torch.bfloat16)
    expected = 100.0 / torch.finfo(FP8_DTYPE).max
    for pos in sorted({0, _BLOCK - 1, _BLOCK, _STRIPE - 1, _STRIPE, numel - 1}):
        if pos >= numel:
            continue
        x[pos] = 100.0
        outputs, descales = fused_per_tensor_fp8_quant(x)
        torch.testing.assert_close(descales[0], x.new_tensor([expected]), rtol=1e-6, atol=0)
        assert outputs[0][pos].float().item() == pytest.approx(448.0, rel=1e-2)
        x[pos] = 0.5


def test_all_zero_and_empty() -> None:
    x = torch.zeros(0, 4, 8, device=DEVICE, dtype=torch.bfloat16)
    outputs, descales = fused_per_tensor_fp8_quant(x)
    assert outputs[0].numel() == 0
    torch.testing.assert_close(descales[0], x.new_tensor([1e-6]), rtol=0, atol=0)

    y = torch.zeros(16, 4, 8, device=DEVICE, dtype=torch.bfloat16)
    outputs, descales = fused_per_tensor_fp8_quant(y)
    assert torch.isfinite(outputs[0].float()).all()
    torch.testing.assert_close(descales[0], y.new_tensor([1e-6]), rtol=0, atol=0)


def test_single_and_two_inputs() -> None:
    a = torch.randn(32, 4, 16, device=DEVICE, dtype=torch.bfloat16)
    b = torch.randn(32, 4, 16, device=DEVICE, dtype=torch.bfloat16)
    for inputs in ((a,), (a, b)):
        outputs, descales = fused_per_tensor_fp8_quant(*inputs)
        for x, x_fp8, descale in zip(inputs, outputs, descales):
            ref_fp8, ref_descale = _reference_quant(x)
            torch.testing.assert_close(descale, ref_descale, rtol=1e-6, atol=0)
            ulp_diff = x_fp8.view(torch.int8).int() - ref_fp8.view(torch.int8).int()
            assert ulp_diff.abs().max() <= 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
