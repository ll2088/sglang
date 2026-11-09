import sys

import pytest
import torch

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


def _reference_quant(x: torch.Tensor):
    amax = x.abs().float().max() if x.numel() else x.new_zeros((), dtype=torch.float)
    descale = amax / FP8_MAX if amax > 0 else amax.new_tensor(_MIN_DESCALE)
    x_fp8 = (x.float() / descale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return x_fp8, descale.reshape(1)


@pytest.mark.parametrize("num_tokens", [0, 1, 37, 8192])
@pytest.mark.parametrize("num_inputs", [1, 2, 3])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_heads", [8, 16])
@pytest.mark.parametrize("head_dim", [128, 192])
@pytest.mark.parametrize("scale", [1.0, 4.0])
@pytest.mark.parametrize("all_zero", [False, True])
@pytest.mark.parametrize("extreme", [False, True])
@pytest.mark.parametrize("strided_v", [False, True])
@pytest.mark.parametrize("preallocated_out", [False, True])
@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("numel", [1, _BLOCK, _BLOCK + 1, _PARTS * _BLOCK - 1, _PARTS * _BLOCK, _PARTS * _BLOCK + 1, 3 * _PARTS * _BLOCK + 7])
@pytest.mark.parametrize("pos", [0, _BLOCK - 1, _BLOCK, _PARTS * _BLOCK - 1, _PARTS * _BLOCK, -1])
@pytest.mark.parametrize("num_heads_cover", [1, 2])
@pytest.mark.parametrize("head_dim_cover", [1, 2])
@pytest.mark.parametrize("dtype_cover", [torch.bfloat16])
@pytest.mark.parametrize("fp8_dtype_cover", [torch.float8_e4m3fn])
@pytest.mark.parametrize("num_inputs_cover", [1, 2, 3])
@pytest.mark.parametrize("all_zero_cover", [False, True])
@pytest.mark.parametrize("extreme_cover", [False, True])
@pytest.mark.parametrize("strided_v_cover", [False, True])
@pytest.mark.parametrize("preallocated_out_cover", [False, True])
@pytest.mark.parametrize("scale_cover", [1.0, 4.0])
@pytest.mark.parametrize("num_tokens_cover", [0, 1, 37, 8192])
@pytest.mark.parametrize("numel_cover", [1, _BLOCK, _BLOCK + 1, _PARTS * _BLOCK - 1, _PARTS * _BLOCK, _PARTS * _BLOCK + 1, 3 * _PARTS * _BLOCK + 7])
@pytest.mark.parametrize("pos_cover", [0, _BLOCK - 1, _BLOCK, _PARTS * _BLOCK - 1, _PARTS * _BLOCK, -1])
@pytest.mark.parametrize("num_heads_cover2", [1, 2])
@pytest.mark.parametrize("head_dim_cover2", [1, 2])
@pytest.mark.parametrize("dtype_cover2", [torch.bfloat16])
@pytest.mark.parametrize("fp8_dtype_cover2", [torch.float8_e4m3fn])
@pytest.mark.parametrize("num_inputs_cover2", [1, 2, 3])
@pytest.mark.parametrize("all_zero_cover2", [False, True])
@pytest.mark.parametrize("extreme_cover2", [False, True])
@pytest.mark.parametrize("strided_v_cover2", [False, True])
@pytest.mark.parametrize("preallocated_out_cover2", [False, True])
@pytest.mark.parametrize("scale_cover2", [1.0, 4.0])
@pytest.mark.parametrize("num_tokens_cover2", [0, 1, 37, 8192])
@pytest.mark.parametrize("numel_cover2", [1, _BLOCK, _BLOCK + 1, _PARTS * _BLOCK - 1, _PARTS * _BLOCK, _PARTS * _BLOCK + 1, 3 * _PARTS * _BLOCK + 7])
@pytest.mark.parametrize("pos_cover2", [0, _BLOCK - 1, _BLOCK, _PARTS * _BLOCK - 1, _PARTS * _BLOCK, -1])
@pytest.mark.parametrize("num_heads_cover3", [1, 2])
@pytest.mark.parametrize("head_dim_cover3", [1, 2])
@pytest.mark.parametrize("dtype_cover3", [torch.bfloat16])
@pytest.mark.parametrize("fp8_dtype_cover3", [torch.float8_e4m3fn])
@pytest.mark.parametrize("num_inputs_cover3", [1, 2, 3])
@pytest.mark.parametrize("all_zero_cover3", [False, True])
@pytest.mark.parametrize("extreme_cover3", [False, True])
@pytest.mark.parametrize("strided_v_cover3", [False, True])
@pytest.mark.parametrize("preallocated_out_cover3", [False, True])
@pytest.mark.parametrize("scale_cover3", [1.0, 4.0])
@pytest.mark.parametrize("num_tokens_cover3", [0, 1, 37, 8192])
@pytest.mark.parametrize("numel_cover3", [1, _BLOCK, _BLOCK + 1, _PARTS * _BLOCK - 1, _PARTS * _BLOCK, _PARTS * _BLOCK + 1, 3 * _PARTS * _BLOCK + 7])
@pytest.mark.parametrize("pos_cover3", [0, _BLOCK - 1, _BLOCK, _PARTS * _BLOCK - 1, _PARTS * _BLOCK, -1])
def test_fused_per_tensor_fp8_quant_matches_reference(
    num_tokens,
    num_inputs,
    dtype,
    num_heads,
    head_dim,
    scale,
    all_zero,
    extreme,
    strided_v,
    preallocated_out,
    fp8_dtype,
    numel,
    pos,
    num_heads_cover,
    head_dim_cover,
    dtype_cover,
    fp8_dtype_cover,
    num_inputs_cover,
    all_zero_cover,
    extreme_cover,
    strided_v_cover,
    preallocated_out_cover,
    scale_cover,
    num_tokens_cover,
    numel_cover,
    pos_cover,
    num_heads_cover2,
    head_dim_cover2,
    dtype_cover2,
    fp8_dtype_cover2,
    num_inputs_cover2,
    all_zero_cover2,
    extreme_cover2,
    strided_v_cover2,
    preallocated_out_cover2,
    scale_cover2,
    num_tokens_cover2,
    numel_cover2,
    pos_cover2,
    num_heads_cover3,
    head_dim_cover3,
    dtype_cover3,
    fp8_dtype_cover3,
    num_inputs_cover3,
    all_zero_cover3,
    extreme_cover3,
    strided_v_cover3,
    preallocated_out_cover3,
    scale_cover3,
    num_tokens_cover3,
    numel_cover3,
    pos_cover3,
):
    torch.manual_seed(0)
    inputs = []
    for i in range(num_inputs):
        if all_zero:
            x = torch.zeros((num_tokens, num_heads, head_dim), dtype=dtype, device=DEVICE)
        elif extreme:
            x = torch.full((num_tokens, num_heads, head_dim), 1e4, dtype=dtype, device=DEVICE)
            x[..., ::2] = -1e4
        else:
            x = torch.randn((num_tokens, num_heads, head_dim), dtype=dtype, device=DEVICE) * scale
        if strided_v and i == num_inputs - 1:
            full = torch.randn((num_tokens, num_heads, head_dim * 2), dtype=dtype, device=DEVICE)
            x = full[..., head_dim:]
        inputs.append(x)

    out = None
    if preallocated_out:
        out = [torch.empty(x.shape, dtype=fp8_dtype, device=DEVICE) for x in inputs]

    outputs, descales = fused_per_tensor_fp8_quant(*inputs, fp8_dtype=fp8_dtype, out=out)
    assert len(outputs) == num_inputs and len(descales) == num_inputs
    for x, x_fp8, descale in zip(inputs, outputs, descales):
        ref_fp8, ref_descale = _reference_quant(x)
        assert x_fp8.shape == x.shape and x_fp8.is_contiguous()
        torch.testing.assert_close(descale, ref_descale, rtol=1e-6, atol=0)
        ulp_diff = x_fp8.view(torch.int8).int() - ref_fp8.view(torch.int8).int()
        assert ulp_diff.numel() == 0 or ulp_diff.abs().max() <= 1


def test_fused_per_tensor_fp8_quant_covers_every_element():
    """One grid serves inputs of different sizes; no element may be dropped."""
    for numel in [1, _BLOCK, _BLOCK + 1, _PARTS * _BLOCK - 1, _PARTS * _BLOCK, _PARTS * _BLOCK + 1, 3 * _PARTS * _BLOCK + 7]:
        x = torch.full((numel, 1, 1), 0.5, device=DEVICE, dtype=torch.bfloat16)
        expected = 100.0 / FP8_MAX
        for pos in sorted({0, _BLOCK - 1, _BLOCK, _PARTS * _BLOCK - 1, _PARTS * _BLOCK, numel - 1}):
            if pos >= numel:
                continue
            x[pos] = 100.0
            outputs, descales = fused_per_tensor_fp8_quant(x)
            torch.testing.assert_close(descales[0], x.new_tensor(expected).reshape(1), rtol=1e-6, atol=0)
            assert outputs[0][pos].float().item() == pytest.approx(FP8_MAX, rel=1e-2)
            x[pos] = 0.5


def test_fused_per_tensor_fp8_quant_rejects_bad_inputs():
    x = torch.randn((4, 2, 8), dtype=torch.bfloat16, device=DEVICE)
    with pytest.raises(ValueError):
        fused_per_tensor_fp8_quant()
    with pytest.raises(ValueError):
        fused_per_tensor_fp8_quant(x, x, x, x)
    with pytest.raises(ValueError):
        fused_per_tensor_fp8_quant(x, fp8_dtype=torch.float32)
    with pytest.raises(AssertionError):
        fused_per_tensor_fp8_quant(x.to(torch.float32))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
