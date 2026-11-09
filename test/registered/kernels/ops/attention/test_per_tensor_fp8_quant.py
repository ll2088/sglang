"""Tests for the fused per-tensor FP8 quantization kernel.

Collectable without a GPU or triton: the module-level skip only fires when a
GPU is present but triton is missing, and every test re-checks via
``_require_gpu``. This keeps the import-smoke and registry checks runnable in
CPU-only CI.
"""

import sys

import pytest

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=60, suite="nightly-amd-kernel-1-gpu", nightly=True)

try:
    import torch
except ImportError:  # pragma: no cover - torch is a hard dep in CI
    torch = None

_HAS_GPU = torch is not None and torch.cuda.is_available()
_HAS_TRITON = False
if _HAS_GPU:
    try:
        import triton  # noqa: F401

        _HAS_TRITON = True
    except ImportError:
        _HAS_TRITON = False

DEVICE = "cuda"
FP8_DTYPE = torch.float8_e4m3fn if torch is not None else None


def _require_gpu() -> None:
    if not _HAS_GPU:
        pytest.skip("per_tensor_fp8_quant requires a CUDA/ROCm device")
    if not _HAS_TRITON:
        pytest.skip("per_tensor_fp8_quant requires triton")


def _kernel():
    from sglang.kernels.ops.attention.per_tensor_fp8_quant import (
        fused_per_tensor_fp8_quant,
    )

    return fused_per_tensor_fp8_quant


def _reference():
    from sglang.kernels.ops.attention.per_tensor_fp8_quant import (
        reference_per_tensor_fp8_quant,
    )

    return reference_per_tensor_fp8_quant


def test_import_smoke() -> None:
    """Importing the module and registry must work without a GPU/triton."""
    import sglang.kernels.ops.attention as attn
    from sglang.kernels.ops.attention import per_tensor_fp8_quant as mod

    assert hasattr(mod, "fused_per_tensor_fp8_quant")
    assert hasattr(mod, "reference_per_tensor_fp8_quant")
    assert "attention.fused_per_tensor_fp8_quant" in attn.registry.ops()
    assert "attention.reference_per_tensor_fp8_quant" in attn.registry.ops()


@pytest.mark.parametrize("num_tokens", [0, 1, 37, 8192])
@pytest.mark.parametrize("num_heads", [1, 12])
@pytest.mark.parametrize("num_inputs", [1, 2, 3])
@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.skipif(not _HAS_GPU, reason="requires a CUDA/ROCm device")
def test_fused_per_tensor_fp8_quant_matches_reference(
    num_tokens: int, num_heads: int, num_inputs: int, fp8_dtype: torch.dtype
) -> None:
    _require_gpu()
    fused = _kernel()
    reference = _reference()
    torch.manual_seed(0)
    qk_nope, qk_rope, v_head = 128, 64, 128
    q = torch.randn(num_tokens, num_heads, qk_nope + qk_rope, device=DEVICE) * 4
    k = torch.randn(num_tokens, num_heads, qk_nope + qk_rope, device=DEVICE)
    kv = torch.randn(num_tokens, num_heads, qk_nope + v_head, device=DEVICE)
    v = kv[..., qk_nope:]
    inputs = [x.to(torch.bfloat16) for x in (q, k, v)][:num_inputs]
    outputs, descales = fused(*inputs, fp8_dtype=fp8_dtype)
    assert len(outputs) == num_inputs and len(descales) == num_inputs
    for x, x_fp8, descale in zip(inputs, outputs, descales):
        ref_fp8, ref_descale = reference(x, fp8_dtype=fp8_dtype)
        assert x_fp8.shape == x.shape and x_fp8.is_contiguous()
        torch.testing.assert_close(descale, ref_descale, rtol=1e-6, atol=0)
        ulp_diff = x_fp8.view(torch.int8).int() - ref_fp8.view(torch.int8).int()
        assert ulp_diff.numel() == 0 or ulp_diff.abs().max() <= 1


@pytest.mark.parametrize("numel", [1, 4096, 4097, 4096 * 256 - 1, 4096 * 256, 4096 * 256 + 1, 3 * 4096 * 256 + 7])
@pytest.mark.skipif(not _HAS_GPU, reason="requires a CUDA/ROCm device")
def test_fused_per_tensor_fp8_quant_covers_every_element(numel: int) -> None:
    _require_gpu()
    fused = _kernel()
    x = torch.full((numel, 1, 1), 0.5, device=DEVICE, dtype=torch.bfloat16)
    expected = 100.0 / torch.finfo(FP8_DTYPE).max
    for pos in sorted({0, 4095, 4096, 4096 * 256 - 1, 4096 * 256, numel - 1}):
        if pos >= numel:
            continue
        x.zero_()
        x.view(-1)[pos] = 100.0
        outputs, descales = fused(x, fp8_dtype=FP8_DTYPE)
        torch.testing.assert_close(descales[0], x.new_tensor([expected]), rtol=1e-6, atol=0)
        assert outputs[0].view(-1)[pos].float().item() == torch.finfo(FP8_DTYPE).max


@pytest.mark.skipif(not _HAS_GPU, reason="requires a CUDA/ROCm device")
def test_fused_per_tensor_fp8_quant_all_zero_and_extreme() -> None:
    _require_gpu()
    fused = _kernel()
    s, num_heads = 16, 4
    x = torch.zeros(s, num_heads, 128, device=DEVICE, dtype=torch.bfloat16)
    outputs, descales = fused(x, fp8_dtype=FP8_DTYPE)
    assert torch.isfinite(outputs[0].float()).all()
    assert descales[0].item() > 0
    x.fill_(1e4)
    outputs, _ = fused(x, fp8_dtype=FP8_DTYPE)
    assert outputs[0].float().abs().max() <= torch.finfo(FP8_DTYPE).max


@pytest.mark.skipif(not _HAS_GPU, reason="requires a CUDA/ROCm device")
def test_fused_per_tensor_fp8_quant_rejects_bad_input() -> None:
    _require_gpu()
    fused = _kernel()
    with pytest.raises(ValueError):
        fused(torch.zeros(4, 4, device=DEVICE, dtype=torch.bfloat16))
    with pytest.raises(ValueError):
        fused(torch.zeros(4, 4, 4, device=DEVICE, dtype=torch.bfloat16), fp8_dtype=torch.float32)


@pytest.mark.skipif(not _HAS_GPU, reason="requires a CUDA/ROCm device")
def test_perf_fused_per_tensor_fp8_quant() -> None:
    """Perf smoke: fused path must not be slower than the torch reference."""
    _require_gpu()
    import time

    fused = _kernel()
    reference = _reference()
    s, h, d = 4096, 12, 192
    x = torch.randn(s, h, d, device=DEVICE, dtype=torch.bfloat16)
    for _ in range(3):
        fused(x, fp8_dtype=FP8_DTYPE)
        reference(x, fp8_dtype=FP8_DTYPE)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        fused(x, fp8_dtype=FP8_DTYPE)
    torch.cuda.synchronize()
    fused_ms = (time.perf_counter() - t0) / 10 * 1e3
    t0 = time.perf_counter()
    for _ in range(10):
        reference(x, fp8_dtype=FP8_DTYPE)
    torch.cuda.synchronize()
    ref_ms = (time.perf_counter() - t0) / 10 * 1e3
    assert fused_ms <= ref_ms * 1.1, (fused_ms, ref_ms)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
