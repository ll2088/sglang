import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=60, suite="nightly-amd-kernel-1-gpu", nightly=True)

if not torch.cuda.is_available():
    pytest.skip("per-tensor FP8 quant requires a CUDA/ROCm device", allow_module_level=True)

try:
    import triton  # noqa: F401
except ImportError:
    pytest.skip("per-tensor FP8 quant requires triton", allow_module_level=True)

from sglang.kernels.ops.attention.per_tensor_fp8_quant import (  # noqa: E402
    fused_per_tensor_fp8_quant,
)

DEVICE = "cuda"
FP8_DTYPE = torch.float8_e4m3fn


def _reference(x: torch.Tensor):
    amax = x.abs().float().max() if x.numel() else x.new_zeros((), dtype=torch.float)
    descale = amax / torch.finfo(FP8_DTYPE).max if amax > 0 else amax.new_tensor(1e-6)
    return (x.float() / descale).clamp(-448, 448).to(FP8_DTYPE), descale.reshape(1)


@pytest.mark.parametrize("num_tokens", [0, 1, 37, 8192])
def test_fused_per_tensor_fp8_quant_matches_reference(num_tokens: int) -> None:
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
        ulp = x_fp8.view(torch.int8).int() - ref_fp8.view(torch.int8).int()
        assert ulp.numel() == 0 or ulp.abs().max() <= 1


def test_fused_per_tensor_fp8_quant_all_zero_and_extreme() -> None:
    x = torch.zeros(16, 4, 128, device=DEVICE, dtype=torch.bfloat16)
    (y,), (s,) = fused_per_tensor_fp8_quant(x)
    assert torch.isfinite(y.float()).all() and s.item() > 0
    x.fill_(1e4)
    (y,), _ = fused_per_tensor_fp8_quant(x)
    assert y.float().abs().max() <= 448


def test_fused_per_tensor_fp8_quant_unaligned_lengths() -> None:
    a = torch.randn(4097, 1, 1, device=DEVICE, dtype=torch.bfloat16)
    b = torch.randn(1, 1, 1, device=DEVICE, dtype=torch.bfloat16)
    (ya, yb), _ = fused_per_tensor_fp8_quant(a, b)
    assert ya.shape == a.shape and yb.shape == b.shape


def test_import_without_triton() -> None:
    import sglang.kernels.ops.attention as m

    assert m is not None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
