import sys

import pytest
import torch

from sglang.kernels.jit.utils import get_ci_test_range
from sglang.kernels.ops.attention.triton_per_tensor_fp8_quant import (
    _BLOCK,
    _MIN_DESCALE,
    _PARTS,
    fused_per_tensor_fp8_quant,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=60, suite="nightly-amd-kernel-1-gpu", nightly=True)

DEVICE = "cuda"
FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = float(torch.finfo(FP8_DTYPE).max)

# Small shapes for CI; the full sweep runs locally.
SHAPES = get_ci_test_range(
    [(1, 1, 1), (1, 8, 128), (37, 12, 192), (8192, 16, 128)],
    [(1, 8, 128), (37, 12, 192)],
)
NUM_INPUTS = get_ci_test_range([1, 2, 3], [1, 3])


def _reference_quant(x: torch.Tensor):
    amax = x.abs().float().max() if x.numel() else x.new_zeros((), dtype=torch.float)
    descale = amax / FP8_MAX if amax > 0 else amax.new_tensor(_MIN_DESCALE)
    x_fp8 = (x.float() / descale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return x_fp8, descale.reshape(1)


def _make_inputs(num_inputs, shape, dtype=torch.bfloat16, scale=1.0):
    torch.manual_seed(0)
    tokens, heads, dim = shape
    return [
        (torch.randn(tokens, heads, dim, device=DEVICE) * scale).to(dtype)
        for _ in range(num_inputs)
    ]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("num_inputs", NUM_INPUTS)
def test_correctness(dtype, shape, num_inputs):
    inputs = _make_inputs(num_inputs, shape, dtype=dtype, scale=4.0)
    outputs, descales = fused_per_tensor_fp8_quant(*inputs)
    assert len(outputs) == num_inputs and len(descales) == num_inputs
    for x, x_fp8, descale in zip(inputs, outputs, descales):
        ref_fp8, ref_descale = _reference_quant(x)
        assert x_fp8.shape == x.shape and x_fp8.is_contiguous()
        assert x_fp8.dtype == FP8_DTYPE
        torch.testing.assert_close(descale, ref_descale, rtol=1e-6, atol=0)
        # x * (1 / descale) may round one e4m3 step away from x / descale.
        ulp_diff = x_fp8.view(torch.int8).int() - ref_fp8.view(torch.int8).int()
        assert ulp_diff.numel() == 0 or ulp_diff.abs().max() <= 1


@pytest.mark.parametrize("num_inputs", [1, 2, 3])
def test_empty_inputs_get_min_descale(num_inputs):
    inputs = [torch.empty(0, 8, 128, device=DEVICE, dtype=torch.bfloat16) for _ in range(num_inputs)]
    outputs, descales = fused_per_tensor_fp8_quant(*inputs)
    for x, x_fp8, descale in zip(inputs, outputs, descales):
        assert x_fp8.shape == x.shape and x_fp8.numel() == 0
        torch.testing.assert_close(descale, torch.full_like(descale, _MIN_DESCALE))


def test_all_zero_inputs_get_min_descale():
    inputs = [torch.zeros(16, 8, 128, device=DEVICE, dtype=torch.bfloat16) for _ in range(3)]
    outputs, descales = fused_per_tensor_fp8_quant(*inputs)
    for x_fp8, descale in zip(outputs, descales):
        assert torch.count_nonzero(x_fp8.view(torch.int8)) == 0
        torch.testing.assert_close(descale, torch.full_like(descale, _MIN_DESCALE))


def test_extreme_values_are_clamped():
    x = torch.full((4, 8, 128), 1e4, device=DEVICE, dtype=torch.bfloat16)
    y = torch.full((4, 8, 128), -1e4, device=DEVICE, dtype=torch.bfloat16)
    outputs, descales = fused_per_tensor_fp8_quant(x, y)
    for x_fp8, descale in zip(outputs, descales):
        assert torch.isfinite(descale).all() and descale.item() > 0
        assert x_fp8.float().abs().max().item() <= FP8_MAX


def test_strided_v_view():
    tokens, heads, nope, v_head = 16, 8, 128, 128
    kv = torch.randn(tokens, heads, nope + v_head, device=DEVICE, dtype=torch.bfloat16)
    v = kv[..., nope:]
    assert v.stride(2) == 1 and not v.is_contiguous()
    q = torch.randn(tokens, heads, nope, device=DEVICE, dtype=torch.bfloat16)
    outputs, descales = fused_per_tensor_fp8_quant(q, v)
    for x, x_fp8, descale in zip((q, v), outputs, descales):
        ref_fp8, ref_descale = _reference_quant(x)
        torch.testing.assert_close(descale, ref_descale, rtol=1e-6, atol=0)
        ulp_diff = x_fp8.view(torch.int8).int() - ref_fp8.view(torch.int8).int()
        assert ulp_diff.numel() == 0 or ulp_diff.abs().max() <= 1


@pytest.mark.parametrize(
    "numel",
    [1, _BLOCK, _BLOCK + 1, _PARTS * _BLOCK - 1, _PARTS * _BLOCK, _PARTS * _BLOCK + 1],
)
def test_covers_every_element(numel):
    x = torch.full((numel, 1, 1), 0.5, device=DEVICE, dtype=torch.bfloat16)
    outputs, descales = fused_per_tensor_fp8_quant(x)
    expected = 0.5 / FP8_MAX
    torch.testing.assert_close(descales[0], torch.full_like(descales[0], expected))
    assert torch.count_nonzero(outputs[0].view(torch.int8)) == numel


def test_out_buffers_are_used():
    inputs = _make_inputs(2, (37, 12, 192))
    outs = [torch.empty_like(x, dtype=FP8_DTYPE) for x in inputs]
    outputs, _ = fused_per_tensor_fp8_quant(*inputs, out=outs)
    assert outputs[0] is outs[0] and outputs[1] is outs[1]


def test_out_shorter_than_inputs_is_padded():
    inputs = _make_inputs(3, (8, 4, 64))
    outs = [torch.empty_like(inputs[0], dtype=FP8_DTYPE)]
    outputs, descales = fused_per_tensor_fp8_quant(*inputs, out=outs)
    assert len(outputs) == 3 and len(descales) == 3
    assert outputs[0] is outs[0]
    assert outputs[1].shape == inputs[1].shape and outputs[2].shape == inputs[2].shape


def test_rejects_bad_input_count():
    x = torch.randn(4, 8, 128, device=DEVICE, dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        fused_per_tensor_fp8_quant()
    with pytest.raises(ValueError):
        fused_per_tensor_fp8_quant(x, x, x, x)


def test_rejects_bad_dtype():
    x = torch.randn(4, 8, 128, device=DEVICE, dtype=torch.float32)
    with pytest.raises(AssertionError):
        fused_per_tensor_fp8_quant(x)


def test_rejects_non_contiguous_inner_dim():
    x = torch.randn(4, 128, 8, device=DEVICE, dtype=torch.bfloat16).transpose(1, 2)
    assert x.stride(2) != 1
    with pytest.raises(AssertionError):
        fused_per_tensor_fp8_quant(x)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
