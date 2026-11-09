import sys

import pytest

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=60, suite="nightly-amd-kernel-1-gpu", nightly=True)

# Import torch/triton lazily so this module is collectable (and skips cleanly)
# on CPU-only or triton-less environments instead of erroring at import time.
try:
    import torch
except ImportError:  # pragma: no cover - torch is a hard dep in CI
    torch = None

_HAS_GPU = torch is not None and torch.cuda.is_available()
if not _HAS_GPU:
    pytest.skip("per-tensor fp8 quant requires a CUDA/ROCm device", allow_module_level=True)
_HAS_TRITON = False
if _HAS_GPU:
    try:
        import triton  # noqa: F401

        _HAS_TRITON = True
    except ImportError:
        _HAS_TRITON = False

DEVICE = "cuda"
FP8_DTYPE = torch.float8_e4m3fn


def _require_gpu() -> None:
    if not _HAS_GPU:
        pytest.skip("per-tensor fp8 quant requires a CUDA/ROCm device")
    if not _HAS_TRITON:
        pytest.skip("per-tensor fp8 quant requires triton")


def _kernel():
    from sglang.kernels.ops.attention.per_tensor_fp8_quant import (
        fused_per_tensor_fp8_quant,
    )

    return fused_per_tensor_fp8_quant


def _reference_quant(x: torch.Tensor):
    amax = x.abs().float().max() if x.numel() else x.new_zeros((), dtype=torch.float)
    descale = amax / torch.finfo(FP8_DTYPE).max if amax > 0 else amax.new_tensor(1e-6)
    x_fp8 = (x.float() / descale).clamp(-448, 448).to(FP8_DTYPE)
    return x_fp8, descale.reshape(1)


@pytest.mark.parametrize("num_tokens", [0, 1, 37, 8192])
def test_fused_per_tensor_fp8_quant_matches_reference(num_tokens: int) -> None:
    _require_gpu()
    fused_per_tensor_fp8_quant = _kernel()
    torch.manual_seed(0)
    q = torch.randn(num_tokens, 12, 192, device=DEVICE) * 4
    k = torch.randn(num_tokens, 12, 192, device=DEVICE)
    kv = torch.randn(num_tokens, 12, 256, device=DEVICE)
    # V is a strided view of the kv_b_proj output, as in MLA prefill.
    v = kv[..., 128:]
    inputs = [x.to(torch.bfloat16) for x in (q, k, v)]
    outputs, descales = fused_per_tensor_fp8_quant(*inputs)
    for x, x_fp8, descale in zip(inputs, outputs, descales):
        ref_fp8, ref_descale = _reference_quant(x)
        assert x_fp8.shape == x.shape and x_fp8.is_contiguous()
        torch.testing.assert_close(descale, ref_descale, rtol=1e-6, atol=0)
        # x * (1 / descale) may round one e4m3 step away from x / descale.
        ulp_diff = x_fp8.view(torch.int8).int() - ref_fp8.view(torch.int8).int()
        assert ulp_diff.numel() == 0 or ulp_diff.abs().max() <= 1


def test_fused_per_tensor_fp8_quant_all_zero_and_extreme() -> None:
    _require_gpu()
    fused_per_tensor_fp8_quant = _kernel()
    s = 16
    zeros = torch.zeros(s, 4, 128, device=DEVICE, dtype=torch.bfloat16)
    outputs, descales = fused_per_tensor_fp8_quant(zeros, zeros, zeros)
    for out, descale in zip(outputs, descales):
        assert torch.isfinite(out.float()).all()
        assert torch.isfinite(descale).all() and (descale > 0).all()

    big = torch.full((s, 4, 128), 1e4, device=DEVICE, dtype=torch.bfloat16)
    outputs, _ = fused_per_tensor_fp8_quant(big, big, big)
    for out in outputs:
        assert out.float().abs().max() <= 448


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
