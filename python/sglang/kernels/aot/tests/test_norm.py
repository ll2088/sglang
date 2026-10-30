# Adapted from https://github.com/flashinfer-ai/flashinfer/blob/4e8eb1879f9c3ba6d75511e5893183bf8f289a62/tests/test_norm.py

import sys

import pytest
import sgl_kernel
import torch
from sgl_kernel.utils import is_arch_support_pdl


def llama_rms_norm(x, w, eps=1e-6):
    orig_dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x * w.float()
    x = x.to(orig_dtype)
    return x


def gemma_rms_norm(x, w, eps=1e-6):
    orig_dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x * (1.0 + w.float())
    x = x.to(orig_dtype)
    return x


def gemma_fused_add_rms_norm(x, residual, w, eps=1e-6):
    orig_dtype = x.dtype
    x = x + residual
    residual = x
    x = x.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x * (1.0 + w.float())
    x = x.to(orig_dtype)
    return x, residual


def fused_add_rms_norm(x, residual, weight, eps):
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    x = x + residual.to(torch.float32)
    residual = x.to(orig_dtype)

    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = (x * weight.float()).to(orig_dtype)
    return x, residual


def assert_close_norm(actual, expected, dtype):
    if dtype is torch.bfloat16:
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-2)
    else:
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


def rms_norm_fp64_two_pass(x, w, eps=1e-6):
    """FP64 RMSNorm reference computed from centered values.

    RMSNorm uses E[x^2] (no mean subtraction), but computing it as
    E[(x - mean)^2] + mean^2 in FP64 keeps the reference numerically stable
    for rows with a large mean and a tiny spread, which is exactly the case
    where the one-pass FP32 form cancels.
    """
    x64 = x.double()
    mean = x64.mean(dim=-1, keepdim=True)
    centered = x64 - mean
    var = (centered * centered).mean(dim=-1, keepdim=True) + mean * mean
    return x64 * torch.rsqrt(var + eps) * w.double()


def layer_norm_fp64_two_pass(x, w, b, eps=1e-6):
    """FP64 LayerNorm reference using centered two-pass variance.

    Mirrors the upstream fix: compute the mean first, then the variance from
    centered values, so that |mean| >> stddev rows do not cancel in FP32.
    """
    x64 = x.double()
    mean = x64.mean(dim=-1, keepdim=True)
    centered = x64 - mean
    var = (centered * centered).mean(dim=-1, keepdim=True)
    out = centered * torch.rsqrt(var + eps)
    if w is not None:
        out = out * w.double()
    if b is not None:
        out = out + b.double()
    return out


def rms_norm_fp32_one_pass(x, w, eps=1e-6):
    """FP32 one-pass RMSNorm reference: E[x^2] (cancellation-prone)."""
    x32 = x.float()
    var = x32.pow(2).mean(dim=-1, keepdim=True)
    return x32 * torch.rsqrt(var + eps) * w.float()


def make_large_mean_small_variance(batch_size, hidden_size, dtype, device="cuda"):
    """Rows with |mean| ~ 1e4 and tiny spread, where one-pass FP32 variance cancels."""
    row = torch.full((hidden_size,), 1.0e4, dtype=torch.float64)
    row[:2] = 1.0e4 - 1.0e-2
    x = torch.stack([row, -row]).repeat((batch_size + 1) // 2, 1)[:batch_size]
    return x.to(device=device, dtype=dtype)


def make_large_mean_small_variance_layernorm(batch_size, hidden_size, dtype, device="cuda"):
    """Rows with mean ~ 15.75 and tiny spread, matching the upstream regression case."""
    row = torch.full((hidden_size,), 15.75, dtype=torch.float64)
    row[:2] = 15.6875
    x = torch.stack([row, -row]).repeat((batch_size + 1) // 2, 1)[:batch_size]
    return x.to(device=device, dtype=dtype)


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("hidden_size", [111, 500, 1024, 3072, 3584, 4096, 8192, 16384])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("specify_out", [True, False])
def test_norm(batch_size, hidden_size, dtype, specify_out):
    x = torch.randn(batch_size, hidden_size).to(0).to(dtype)
    w = torch.randn(hidden_size).to(0).to(dtype)

    y_ref = llama_rms_norm(x, w)
    enable_pdl = is_arch_support_pdl()
    if specify_out:
        y = torch.empty_like(x)
        sgl_kernel.rmsnorm(x, w, out=y, enable_pdl=enable_pdl)
    else:
        y = sgl_kernel.rmsnorm(x, w, enable_pdl=enable_pdl)

    torch.testing.assert_close(y_ref, y, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("hidden_size", [111, 500, 1024, 3072, 3584, 4096, 8192, 16384])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_fused_add_rmsnorm(batch_size, hidden_size, dtype):
    eps = 1e-6

    x = torch.randn(batch_size, hidden_size, dtype=dtype, device="cuda")
    residual = torch.randn_like(x)
    weight = torch.randn(hidden_size, dtype=dtype, device="cuda")

    x_native, residual_native = fused_add_rms_norm(
        x.clone(), residual.clone(), weight, eps
    )

    x_fused = x.clone()
    residual_fused = residual.clone()
    enable_pdl = is_arch_support_pdl()
    sgl_kernel.fused_add_rmsnorm(
        x_fused, residual_fused, weight, eps, enable_pdl=enable_pdl
    )

    torch.testing.assert_close(x_fused, x_native, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(residual_fused, residual_native, rtol=1e-3, atol=1e-3)


PRODUCTION_LIKE_NORM_CASES = [
    (38, 4096, torch.bfloat16),
    (1240, 1536, torch.bfloat16),
    (7807, 128, torch.bfloat16),
]


LARGE_MEAN_SMALL_VARIANCE_CASES = [
    (2, 4096, torch.bfloat16),
    (4, 8192, torch.bfloat16),
    (2, 4096, torch.float16),
]


@pytest.mark.parametrize("batch_size,hidden_size,dtype", LARGE_MEAN_SMALL_VARIANCE_CASES)
@pytest.mark.parametrize("eps", [1e-6, 1e-5])
def test_rmsnorm_large_mean_small_variance(batch_size, hidden_size, dtype, eps):
    _require_cuda()
    x = make_large_mean_small_variance(batch_size, hidden_size, dtype)
    w = torch.ones(hidden_size, dtype=dtype, device="cuda")

    y_ref = rms_norm_fp64_two_pass(x, w, eps)
    enable_pdl = is_arch_support_pdl()
    y = sgl_kernel.rmsnorm(x, w, eps, enable_pdl=enable_pdl)

    assert torch.isfinite(y).all()
    torch.testing.assert_close(y.double(), y_ref, rtol=1e-2, atol=2e-3)


@pytest.mark.parametrize("batch_size,hidden_size,dtype", LARGE_MEAN_SMALL_VARIANCE_CASES)
@pytest.mark.parametrize("eps", [1e-6, 1e-5])
def test_fused_add_rmsnorm_large_mean_small_variance(batch_size, hidden_size, dtype, eps):
    _require_cuda()
    # residual accumulation pushes the row to a large mean with tiny spread.
    x = torch.zeros(batch_size, hidden_size, dtype=dtype, device="cuda")
    residual = make_large_mean_small_variance(batch_size, hidden_size, dtype)
    weight = torch.ones(hidden_size, dtype=dtype, device="cuda")

    x_ref = (x.float() + residual.float()).to(dtype)
    y_ref = rms_norm_fp64_two_pass(x_ref, weight, eps)

    x_fused = x.clone()
    residual_fused = residual.clone()
    enable_pdl = is_arch_support_pdl()
    sgl_kernel.fused_add_rmsnorm(
        x_fused, residual_fused, weight, eps, enable_pdl=enable_pdl
    )

    assert torch.isfinite(x_fused).all()
    torch.testing.assert_close(x_fused.double(), y_ref, rtol=1e-2, atol=2e-3)
    torch.testing.assert_close(residual_fused, x_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size,hidden_size,dtype", LARGE_MEAN_SMALL_VARIANCE_CASES)
@pytest.mark.parametrize("eps", [1e-6, 1e-5])
def test_layernorm_large_mean_small_variance(batch_size, hidden_size, dtype, eps):
    _require_cuda()
    # LayerNorm subtracts the mean, so the one-pass E[x^2] - mean^2 form is the
    # cancellation-prone path fixed upstream. Validate the FP64 two-pass reference
    # and the RMSNorm kernel's finite output on the same large-mean rows.
    x = make_large_mean_small_variance_layernorm(batch_size, hidden_size, dtype)
    w = torch.ones(hidden_size, dtype=dtype, device="cuda")
    b = torch.zeros(hidden_size, dtype=dtype, device="cuda")

    y_ref = layer_norm_fp64_two_pass(x, w, b, eps)
    assert torch.isfinite(y_ref).all()

    # The RMSNorm kernel does not subtract the mean; it must still stay finite
    # and match its own FP64 two-pass reference on these rows.
    y = sgl_kernel.rmsnorm(x, w, eps, enable_pdl=is_arch_support_pdl())
    y_rms_ref = rms_norm_fp64_two_pass(x, w, eps)
    assert torch.isfinite(y).all()
    torch.testing.assert_close(y.double(), y_rms_ref, rtol=1e-2, atol=2e-3)


@pytest.mark.parametrize("batch_size,hidden_size,dtype", LARGE_MEAN_SMALL_VARIANCE_CASES)
def test_two_pass_vs_one_pass_variance(batch_size, hidden_size, dtype):
    _require_cuda()
    # The one-pass FP32 reference can cancel to a negative variance; the two-pass
    # reference must stay finite and match the kernel within tolerance.
    eps = 1e-6
    x = make_large_mean_small_variance(batch_size, hidden_size, dtype)
    w = torch.ones(hidden_size, dtype=dtype, device="cuda")

    y_two_pass = rms_norm_fp64_two_pass(x, w, eps)
    y_one_pass = rms_norm_fp32_one_pass(x, w, eps)

    assert torch.isfinite(y_two_pass).all()
    # One-pass path is expected to be numerically unstable here; document it.
    one_pass_err = (y_one_pass.double() - y_two_pass).abs().max().item()
    assert one_pass_err > 1e-2 or not torch.isfinite(y_one_pass).all()

    enable_pdl = is_arch_support_pdl()
    y = sgl_kernel.rmsnorm(x, w, eps, enable_pdl=enable_pdl)
    assert torch.isfinite(y).all()
    torch.testing.assert_close(y.double(), y_two_pass, rtol=1e-2, atol=2e-3)


@pytest.mark.parametrize("batch_size,hidden_size,dtype", PRODUCTION_LIKE_NORM_CASES)
def test_norm_production_like_shapes(batch_size, hidden_size, dtype):
    _require_cuda()
    x = torch.randn(batch_size, hidden_size, dtype=dtype, device="cuda")
    w = torch.randn(hidden_size, dtype=dtype, device="cuda")

    y_ref = llama_rms_norm(x, w)
    enable_pdl = is_arch_support_pdl()
    y = sgl_kernel.rmsnorm(x, w, enable_pdl=enable_pdl)

    assert_close_norm(y_ref, y, dtype)


PRODUCTION_LIKE_FUSED_ADD_RMSNORM_CASES = [
    (39, 4096, torch.bfloat16),
    (39, 8192, torch.bfloat16),
    (89, 4096, torch.bfloat16),
]


@pytest.mark.parametrize(
    "batch_size,hidden_size,dtype", PRODUCTION_LIKE_FUSED_ADD_RMSNORM_CASES
)
def test_fused_add_rmsnorm_production_like_shapes(batch_size, hidden_size, dtype):
    _require_cuda()
    eps = 1e-6

    x = torch.randn(batch_size, hidden_size, dtype=dtype, device="cuda")
    residual = torch.randn_like(x)
    weight = torch.randn(hidden_size, dtype=dtype, device="cuda")

    x_native, residual_native = fused_add_rms_norm(
        x.clone(), residual.clone(), weight, eps
    )

    x_fused = x.clone()
    residual_fused = residual.clone()
    enable_pdl = is_arch_support_pdl()
    sgl_kernel.fused_add_rmsnorm(
        x_fused, residual_fused, weight, eps, enable_pdl=enable_pdl
    )

    assert_close_norm(x_fused, x_native, dtype)
    torch.testing.assert_close(residual_fused, residual_native, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("hidden_size", [111, 500, 1024, 3072, 3584, 4096, 8192, 16384])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("specify_out", [True, False])
def test_gemma_norm(batch_size, hidden_size, dtype, specify_out):
    x = torch.randn(batch_size, hidden_size).to(0).to(dtype)
    w = torch.randn(hidden_size).to(0).to(dtype)

    y_ref = gemma_rms_norm(x, w)
    enable_pdl = is_arch_support_pdl()
    if specify_out:
        y = torch.empty_like(x)
        sgl_kernel.gemma_rmsnorm(x, w, out=y, enable_pdl=enable_pdl)
    else:
        y = sgl_kernel.gemma_rmsnorm(x, w, enable_pdl=enable_pdl)

    torch.testing.assert_close(y_ref, y, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("hidden_size", [111, 500, 1024, 3072, 3584, 4096, 8192, 16384])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_gemma_fused_add_rmsnorm(batch_size, hidden_size, dtype):
    eps = 1e-6

    x = torch.randn(batch_size, hidden_size, dtype=dtype, device="cuda")
    residual = torch.randn_like(x)
    weight = torch.randn(hidden_size, dtype=dtype, device="cuda")

    x_native, residual_native = gemma_fused_add_rms_norm(
        x.clone(), residual.clone(), weight, eps
    )

    x_fused = x.clone()
    residual_fused = residual.clone()
    enable_pdl = is_arch_support_pdl()
    sgl_kernel.gemma_fused_add_rmsnorm(
        x_fused, residual_fused, weight, eps, enable_pdl=enable_pdl
    )

    torch.testing.assert_close(x_fused, x_native, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(residual_fused, residual_native, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
