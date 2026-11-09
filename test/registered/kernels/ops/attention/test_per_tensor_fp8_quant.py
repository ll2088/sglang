import sys

import pytest

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=60, suite="nightly-amd-kernel-1-gpu", nightly=True)

try:
    import torch
except ImportError:  # pragma: no cover
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
        reference_per_tensor_fp8_quant,
    )

    return fused_per_tensor_fp8_quant, reference_per_tensor_fp8_quant


def test_import_smoke() -> None:
    import sglang.kernels.ops.attention as m  # noqa: F401

    from sglang.kernels.registry import registry

    assert registry.has("attention.fused_per_tensor_fp8_quant")


@pytest.mark.parametrize("num_tokens", [0, 1, 37, 8192])
@pytest.mark.parametrize("num_inputs", [1, 2, 3])
@pytest.mark.parametrize("num_heads", [1, 12])
def test_fused_per_tensor_fp8_quant_matches_reference(
    num_tokens: int, num_inputs: int, num_heads: int
) -> None:
    _require_gpu()
    fused, reference = _kernel()
    torch.manual_seed(0)
    qk_nope, qk_rope, v_head = 128, 64, 128
    kv = torch.randn(
        num_tokens, num_heads, qk_nope + v_head, device=DEVICE, dtype=torch.bfloat16
    )
    candidates = [
        torch.randn(num_tokens, num_heads, qk_nope + qk_rope, device=DEVICE, dtype=torch.bfloat16),
        torch.randn(num_tokens, num_heads, qk_nope + qk_rope, device=DEVICE, dtype=torch.bfloat16),
        kv[..., qk_nope:],
    ]
    inputs = candidates[:num_inputs]
    outputs, descales = fused(*inputs, fp8_dtype=FP8_DTYPE)
    ref_outputs, ref_descales = reference(*inputs, fp8_dtype=FP8_DTYPE)
    assert len(outputs) == len(inputs) == len(descales)
    for x, y, d, ry, rd in zip(inputs, outputs, descales, ref_outputs, ref_descales):
        assert y.shape == x.shape and y.is_contiguous()
        torch.testing.assert_close(d, rd, rtol=1e-6, atol=0)
        ulp = y.view(torch.int8).int() - ry.view(torch.int8).int()
        assert ulp.numel() == 0 or ulp.abs().max() <= 1


def test_fused_per_tensor_fp8_quant_all_zero_and_extreme() -> None:
    _require_gpu()
    fused, _ = _kernel()
    s, h = 16, 4
    zeros = torch.zeros(s, h, 128, device=DEVICE, dtype=torch.bfloat16)
    outs, descales = fused(zeros, zeros, zeros, fp8_dtype=FP8_DTYPE)
    for y, d in zip(outs, descales):
        assert torch.isfinite(y.float()).all()
        assert float(d) > 0
    big = torch.full((s, h, 128), 1e4, device=DEVICE, dtype=torch.bfloat16)
    outs, _ = fused(big, -big, big, fp8_dtype=FP8_DTYPE)
    for y in outs:
        assert y.float().abs().max() <= 448


def test_fused_per_tensor_fp8_quant_covers_every_element() -> None:
    _require_gpu()
    fused, _ = _kernel()
    from sglang.kernels.ops.attention.per_tensor_fp8_quant import _BLOCK, _PARTS

    stripe = _PARTS * _BLOCK
    for numel in (1, _BLOCK, _BLOCK + 1, stripe - 1, stripe, stripe + 1, 3 * stripe + 7):
        x = torch.full((numel, 1, 1), 0.5, device=DEVICE, dtype=torch.bfloat16)
        outs, descales = fused(x, fp8_dtype=FP8_DTYPE)
        expected = 0.5 / (100.0 / 448.0)
        assert float(descales[0]) == pytest.approx(100.0 / 448.0, rel=1e-6)
        assert outs[0].float().max().item() == pytest.approx(expected, rel=1e-2)


def test_fused_per_tensor_fp8_quant_rejects_bad_input() -> None:
    _require_gpu()
    fused, _ = _kernel()
    with pytest.raises(ValueError):
        fused(torch.zeros(4, 4, device=DEVICE, dtype=torch.bfloat16))
    with pytest.raises(ValueError):
        fused(torch.zeros(4, 4, 4, device=DEVICE, dtype=torch.bfloat16), fp8_dtype=torch.float32)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
