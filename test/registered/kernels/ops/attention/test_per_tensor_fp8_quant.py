"""Tests for per-tensor FP8 quantization used by MLA prefill.

These tests are collectable without a GPU or triton: the pure-Python reference
runs everywhere, and the GPU kernel tests skip via ``pytest.importorskip``.
"""

import sys

import pytest

from sglang.kernels.ops.attention.per_tensor_fp8_quant import (
    FP8_MAX,
    MIN_DESCALE,
    reference_per_tensor_fp8_quant,
)


def test_reference_quant_basic():
    q, descale = reference_per_tensor_fp8_quant([1.0, -2.0, 4.0])
    assert descale == pytest.approx(4.0 / FP8_MAX)
    assert max(abs(v) for v in q) <= FP8_MAX


def test_reference_quant_all_zero_and_empty():
    for values in ([], [0.0, 0.0]):
        q, descale = reference_per_tensor_fp8_quant(values)
        assert descale == MIN_DESCALE
        assert all(v == 0.0 for v in q)


def test_reference_quant_extreme_values():
    q, _ = reference_per_tensor_fp8_quant([1e9, -1e9])
    assert max(abs(v) for v in q) <= FP8_MAX


def test_import_smoke():
    import sglang.kernels.ops.attention as m

    assert m is not None


def test_gpu_kernel_matches_reference():
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA/ROCm device")
    from sglang.kernels.ops.attention.per_tensor_fp8_quant import (
        fused_per_tensor_fp8_quant,
    )

    x = torch.randn(37, 12, 192, device="cuda", dtype=torch.bfloat16) * 4
    (out,), (scale,) = fused_per_tensor_fp8_quant(x)
    ref_q, ref_scale = reference_per_tensor_fp8_quant(x.flatten().tolist())
    assert scale.item() == pytest.approx(ref_scale, rel=1e-6)
    assert out.shape == x.shape


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
