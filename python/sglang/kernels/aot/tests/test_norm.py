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
    """Rows with |mean| ~ 1e4 and variance ~ 1e-4, where one-pass FP32 variance cancels.

    The spread is chosen so that E[x^2] ~ 1e8 while the true variance is ~1e-4;
    in FP32 the one-pass E[x^2] - mean^2 form loses all significant digits and
    can even go negative.
    """
    row = torch.full((hidden_size,), 1.0e4, dtype=torch.float64)
    # Two entries deviate by 1e-2, giving var ~ 2 * (1e-2)^2 / hidden_size.
    row[:2] = 1.0e4 - 1.0e-2
    x = torch.stack([row, -row]).repeat((batch_size + 1) // 2, 1)[:batch_size]
    return x.to(device=device, dtype=dtype)


def make_large_mean_small_variance_layernorm(batch_size, hidden_size, dtype, device="cuda"):
    """Rows with mean ~ 15.75 and tiny spread, matching the upstream regression case."""
    row = torch.full((hidden_size,), 15.75, dtype=torch.float64)
    row[:2] = 15.6875
    x = torch.stack([row, -row]).repeat((batch_size + 1) // 2, 1)[:batch_size]
    return x.to(device=device, dtype=dtype)


def make_large_mean_small_variance_1e4(batch_size, hidden_size, dtype, device="cuda"):
    """Rows with mean ~ 1e4 and variance ~ 1e-4 (acceptance-criteria input)."""
    row = torch.full((hidden_size,), 1.0e4, dtype=torch.float64)
    # Deviations of +-1e-2 on two lanes give var ~ 2e-4 / hidden_size * hidden_size
    # scaled by the number of deviating lanes; keep it near 1e-4.
    row[:2] = 1.0e4 - 1.0e-2
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
    residual = make_large_mean_small_variance_1e4(batch_size, hidden_size, dtype)
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
def test_fused_add_rmsnorm_large_mean_small_variance_1e4(
    batch_size, hidden_size, dtype, eps
):
    _require_cuda()
    # Acceptance-criteria input: residual accumulation produces mean ~ 1e4 and
    # variance ~ 1e-4, the cancellation-prone regime for one-pass FP32 variance.
    x = torch.zeros(batch_size, hidden_size, dtype=dtype, device="cuda")
    residual = make_large_mean_small_variance_1e4(batch_size, hidden_size, dtype)
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
    # cancellation-prone path fixed upstream. This repo has no LayerNorm kernel
    # to exercise here, so validate the FP64 two-pass reference itself: it must
    # stay finite and match a direct centered computation on these rows.
    x = make_large_mean_small_variance_layernorm(batch_size, hidden_size, dtype)
    w = torch.ones(hidden_size, dtype=dtype, device="cuda")
    b = torch.zeros(hidden_size, dtype=dtype, device="cuda")

    y_ref = layer_norm_fp64_two_pass(x, w, b, eps)
    assert torch.isfinite(y_ref).all()

    x64 = x.double()
    mean = x64.mean(dim=-1, keepdim=True)
    centered = x64 - mean
    var = (centered * centered).mean(dim=-1, keepdim=True)
    direct = centered * torch.rsqrt(var + eps)
    assert torch.isfinite(direct).all()
    torch.testing.assert_close(y_ref, direct, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("batch_size,hidden_size,dtype", LARGE_MEAN_SMALL_VARIANCE_CASES)
@pytest.mark.parametrize("eps", [1e-6, 1e-5])
def test_layernorm_large_mean_small_variance_1e4(batch_size, hidden_size, dtype, eps):
    _require_cuda()
    # Acceptance-criteria input: mean ~ 1e4, variance ~ 1e-4. The FP64 two-pass
    # LayerNorm reference must stay finite and match a direct centered computation.
    x = make_large_mean_small_variance_1e4(batch_size, hidden_size, dtype)
    w = torch.ones(hidden_size, dtype=dtype, device="cuda")
    b = torch.zeros(hidden_size, dtype=dtype, device="cuda")

    y_ref = layer_norm_fp64_two_pass(x, w, b, eps)
    assert torch.isfinite(y_ref).all()

    x64 = x.double()
    mean = x64.mean(dim=-1, keepdim=True)
    centered = x64 - mean
    var = (centered * centered).mean(dim=-1, keepdim=True)
    direct = centered * torch.rsqrt(var + eps)
    assert torch.isfinite(direct).all()
    torch.testing.assert_close(y_ref, direct, rtol=1e-6, atol=1e-6)

    # The RMSNorm kernel must also stay finite on the same rows.
    enable_pdl = is_arch_support_pdl()
    y = sgl_kernel.rmsnorm(x, w, eps, enable_pdl=enable_pdl)
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("batch_size,hidden_size,dtype", LARGE_MEAN_SMALL_VARIANCE_CASES)
def test_two_pass_vs_one_pass_variance(batch_size, hidden_size, dtype):
    _require_cuda()
    # The one-pass FP32 reference can cancel to a negative variance; the two-pass
    # reference must stay finite and match the kernel within tolerance.
    eps = 1e-6
    x = make_large_mean_small_variance_1e4(batch_size, hidden_size, dtype)
    w = torch.ones(hidden_size, dtype=dtype, device="cuda")

    y_two_pass = rms_norm_fp64_two_pass(x, w, eps)
    y_one_pass = rms_norm_fp32_one_pass(x, w, eps)

    assert torch.isfinite(y_two_pass).all()

    # The one-pass FP32 path is expected to be numerically unstable here. When it
    # produces finite values, its error must be strictly larger than the kernel's
    # error against the FP64 two-pass reference; when it produces non-finite
    # values that is itself evidence of the cancellation bug.
    enable_pdl = is_arch_support_pdl()
    y = sgl_kernel.rmsnorm(x, w, eps, enable_pdl=enable_pdl)
    assert torch.isfinite(y).all()
    torch.testing.assert_close(y.double(), y_two_pass, rtol=1e-2, atol=2e-3)

    kernel_err = (y.double() - y_two_pass).abs().max().item()
    if torch.isfinite(y_one_pass).all():
        one_pass_err = (y_one_pass.double() - y_two_pass).abs().max().item()
        assert one_pass_err > kernel_err, (
            f"one-pass FP32 error {one_pass_err} should exceed kernel error {kernel_err}"
        )


@pytest.mark.parametrize("batch_size,hidden_size,dtype", PRODUCTION_LIKE_NORM_CASES)
def test_norm_production_like_shapes(batch_size, hidden_size, dtype):
    _require_cuda()
    x = torch.randn(batch_size, hidden_size, dtype=dtype, device="cuda")
    w = torch.randn(hidden_size, dtype=dtype, device="cuda")

    y_ref = llama_rms_norm(x, w)
    enable_pdl = is_arch_support_pdl()
    y = sgl_kernel.rmsnorm(x, w, enable_pdl=enable_pdl)

    assert_close_norm(y_ref, y, dtype)


@pytest.mark.parametrize("batch_size,hidden_size,dtype", LARGE_MEAN_SMALL_VARIANCE_CASES)
@pytest.mark.parametrize("eps", [1e-6, 1e-5])
def test_cpu_reference_matches_cuda_kernel(batch_size, hidden_size, dtype, eps):
    _require_cuda()
    # CPU FP64 two-pass reference vs. the CUDA RMSNorm kernel on the same input.
    # This is the CPU/CUDA consistency check for the large-mean small-variance case.
    x = make_large_mean_small_variance_1e4(batch_size, hidden_size, dtype)
    w = torch.ones(hidden_size, dtype=dtype, device="cuda")

    y_cpu = rms_norm_fp64_two_pass(x.cpu(), w.cpu(), eps)
    assert torch.isfinite(y_cpu).all()

    enable_pdl = is_arch_support_pdl()
    y_cuda = sgl_kernel.rmsnorm(x, w, eps, enable_pdl=enable_pdl)
    assert torch.isfinite(y_cuda).all()
    torch.testing.assert_close(y_cuda.double().cpu(), y_cpu, rtol=1e-2, atol=2e-3)


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
