import sys

import pytest
import torch

from sglang.kernels.ops.attention.triton_per_tensor_fp8_quant import (
    _MIN_DESCALE,
    fused_per_tensor_fp8_quant,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=60, suite="nightly-amd-kernel-1-gpu", nightly=True)

DEVICE = "cuda"
FP8_DTYPE = torch.float8_e4m3fn


def _has_cuda() -> bool:
    return torch.cuda.is_available()


pytestmark = pytest.mark.skipif(
    not _has_cuda(), reason="fused_per_tensor_fp8_quant requires a CUDA/ROCm device"
)


def _reference_quant(x: torch.Tensor, fp8_dtype: torch.dtype = FP8_DTYPE):
    fp8_max = float(torch.finfo(fp8_dtype).max)
    if x.numel() == 0:
        amax = x.new_zeros((), dtype=torch.float32)
    else:
        amax = x.abs().float().max()
    descale = torch.maximum(
        amax / fp8_max, amax.new_tensor(_MIN_DESCALE)
    ).reshape(1)
    y = (x.float() / descale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    return y, descale


@pytest.mark.parametrize("num_tokens", [0, 1, 37, 8192])
@pytest.mark.parametrize("num_inputs", [1, 2, 3])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_correctness(num_tokens, num_inputs, dtype, fp8_dtype):
    torch.manual_seed(0)
    heads, dim = 4, 128
    inputs = [
        (torch.randn(num_tokens, heads, dim, device=DEVICE) * (i + 1)).to(dtype)
        for i in range(num_inputs)
    ]
    outputs, descales = fused_per_tensor_fp8_quant(*inputs, fp8_dtype=fp8_dtype)
    assert len(outputs) == num_inputs and len(descales) == num_inputs
    for x, y, d in zip(inputs, outputs, descales):
        ref_y, ref_d = _reference_quant(x, fp8_dtype)
        assert y.shape == x.shape and y.is_contiguous()
        assert y.dtype == fp8_dtype
        torch.testing.assert_close(d, ref_d, rtol=1e-6, atol=0)
        # x * (1 / descale) may round one e4m3 step away from x / descale.
        ulp = y.view(torch.int8).int() - ref_y.view(torch.int8).int()
        assert ulp.numel() == 0 or ulp.abs().max() <= 1


def test_strided_v_view_matches_reference():
    torch.manual_seed(0)
    num_tokens, heads, nope, v_head = 64, 8, 128, 128
    kv = torch.randn(num_tokens, heads, nope + v_head, device=DEVICE).to(
        torch.bfloat16
    )
    v = kv[..., nope:]
    assert v.stride(-1) == 1
    (v_fp8,), (descale,) = fused_per_tensor_fp8_quant(v)
    ref_v, ref_d = _reference_quant(v)
    torch.testing.assert_close(descale, ref_d, rtol=1e-6, atol=0)
    ulp = v_fp8.view(torch.int8).int() - ref_v.view(torch.int8).int()
    assert ulp.abs().max() <= 1


@pytest.mark.parametrize("num_tokens", [0, 1, 7])
def test_empty_and_zero_inputs_get_min_descale(num_tokens):
    x = torch.zeros(num_tokens, 2, 64, device=DEVICE, dtype=torch.bfloat16)
    (y,), (d,) = fused_per_tensor_fp8_quant(x)
    assert y.shape == x.shape
    torch.testing.assert_close(d, torch.full_like(d, _MIN_DESCALE))
    assert torch.isfinite(y.float()).all()


def test_extreme_values_are_clamped():
    x = torch.full((16, 2, 64), 1e4, device=DEVICE, dtype=torch.bfloat16)
    x[0, 0, 0] = -1e4
    (y,), (d,) = fused_per_tensor_fp8_quant(x)
    fp8_max = float(torch.finfo(FP8_DTYPE).max)
    assert y.float().abs().max() <= fp8_max
    assert torch.isfinite(d).all() and d.item() > 0


def test_out_buffer_reuse():
    x = torch.randn(32, 4, 128, device=DEVICE, dtype=torch.bfloat16)
    out = torch.empty_like(x, dtype=FP8_DTYPE)
    (y,), _ = fused_per_tensor_fp8_quant(x, out=[out])
    assert y.data_ptr() == out.data_ptr()


def test_invalid_inputs_raise():
    x = torch.randn(4, 2, 64, device=DEVICE, dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        fused_per_tensor_fp8_quant(*([x] * 4))
    with pytest.raises(ValueError):
        fused_per_tensor_fp8_quant(x, fp8_dtype=torch.float32)
    with pytest.raises(AssertionError):
        fused_per_tensor_fp8_quant(x.transpose(1, 2))


def test_import_and_api_surface():
    import sglang.kernels.ops.attention.triton_per_tensor_fp8_quant as mod

    assert callable(mod.fused_per_tensor_fp8_quant)
    assert mod._MIN_DESCALE > 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
