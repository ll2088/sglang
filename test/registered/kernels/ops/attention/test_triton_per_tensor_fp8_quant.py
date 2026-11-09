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

NUM_TOKENS = get_ci_test_range(
    [0, 1, 37, 8192],
    [0, 1, 37],
)
NUM_HEADS = get_ci_test_range([1, 8, 16], [1, 8])
HEAD_DIM = get_ci_test_range([64, 128], [128])


def _reference_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    amax = x.abs().float().max() if x.numel() else x.new_zeros((), dtype=torch.float)
    descale = amax / FP8_MAX if amax > 0 else amax.new_tensor(_MIN_DESCALE)
    x_fp8 = (x.float() / descale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return x_fp8, descale.reshape(1)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_dim", HEAD_DIM)
def test_fused_per_tensor_fp8_quant_matches_reference(
    num_tokens: int, num_heads: int, head_dim: int
) -> None:
    torch.manual_seed(0)
    q = torch.randn(num_tokens, num_heads, head_dim, device=DEVICE) * 4
    k = torch.randn(num_tokens, num_heads, head_dim, device=DEVICE)
    kv = torch.randn(num_tokens, num_heads, head_dim * 2, device=DEVICE)
    # V is a strided view of the kv_b_proj output, as in MLA prefill.
    v = kv[..., head_dim:]
    inputs = [x.to(torch.bfloat16) for x in (q, k, v)]
    outputs, descales = fused_per_tensor_fp8_quant(*inputs)
    assert len(outputs) == len(inputs) == len(descales)
    for x, x_fp8, descale in zip(inputs, outputs, descales):
        ref_fp8, ref_descale = _reference_quant(x)
        assert x_fp8.shape == x.shape and x_fp8.is_contiguous()
        assert x_fp8.dtype == FP8_DTYPE
        torch.testing.assert_close(descale, ref_descale, rtol=1e-6, atol=0)
        # x * (1 / descale) may round one e4m3 step away from x / descale.
        ulp_diff = x_fp8.view(torch.int8).int() - ref_fp8.view(torch.int8).int()
        assert ulp_diff.numel() == 0 or ulp_diff.abs().max() <= 1


@pytest.mark.parametrize("num_inputs", [1, 2, 3])
def test_fused_per_tensor_fp8_quant_arity(num_inputs: int) -> None:
    torch.manual_seed(0)
    inputs = [
        torch.randn(16, 4, 128, device=DEVICE, dtype=torch.bfloat16)
        for _ in range(num_inputs)
    ]
    outputs, descales = fused_per_tensor_fp8_quant(*inputs)
    assert len(outputs) == num_inputs
    assert len(descales) == num_inputs
    for x, x_fp8, descale in zip(inputs, outputs, descales):
        ref_fp8, ref_descale = _reference_quant(x)
        torch.testing.assert_close(descale, ref_descale, rtol=1e-6, atol=0)
        ulp_diff = x_fp8.view(torch.int8).int() - ref_fp8.view(torch.int8).int()
        assert ulp_diff.numel() == 0 or ulp_diff.abs().max() <= 1


@pytest.mark.parametrize("num_tokens", [0, 1, 37])
def test_fused_per_tensor_fp8_quant_all_zero(num_tokens: int) -> None:
    x = torch.zeros(num_tokens, 8, 128, device=DEVICE, dtype=torch.bfloat16)
    outputs, descales = fused_per_tensor_fp8_quant(x)
    assert outputs[0].shape == x.shape
    assert outputs[0].dtype == FP8_DTYPE
    torch.testing.assert_close(
        descales[0], torch.full_like(descales[0], _MIN_DESCALE), rtol=0, atol=0
    )
    assert torch.count_nonzero(outputs[0].float()) == 0


def test_fused_per_tensor_fp8_quant_extreme_values() -> None:
    x = torch.full((4, 2, 64), 1e4, device=DEVICE, dtype=torch.bfloat16)
    outputs, descales = fused_per_tensor_fp8_quant(x)
    ref_fp8, ref_descale = _reference_quant(x)
    torch.testing.assert_close(descales[0], ref_descale, rtol=1e-6, atol=0)
    ulp_diff = outputs[0].view(torch.int8).int() - ref_fp8.view(torch.int8).int()
    assert ulp_diff.numel() == 0 or ulp_diff.abs().max() <= 1


_STRIPE = _PARTS * _BLOCK


@pytest.mark.parametrize(
    "numel",
    [1, _BLOCK, _BLOCK + 1, _STRIPE - 1, _STRIPE, _STRIPE + 1, 3 * _STRIPE + 7],
)
def test_fused_per_tensor_fp8_quant_covers_every_element(numel: int) -> None:
    """One grid serves inputs of different sizes; no element may be dropped."""
    x = torch.full((numel, 1, 1), 0.5, device=DEVICE, dtype=torch.bfloat16)
    expected = 100.0 / FP8_MAX
    for pos in sorted({0, _BLOCK - 1, _BLOCK, _STRIPE - 1, _STRIPE, numel - 1}):
        if pos >= numel:
            continue
        x[pos] = 100.0
        outputs, descales = fused_per_tensor_fp8_quant(x)
        torch.testing.assert_close(
            descales[0], torch.full_like(descales[0], expected), rtol=1e-6, atol=0
        )
        assert outputs[0][pos].float().item() == pytest.approx(FP8_MAX, rel=1e-2)
        x[pos] = 0.5


def test_fused_per_tensor_fp8_quant_rejects_bad_arity() -> None:
    with pytest.raises(ValueError):
        fused_per_tensor_fp8_quant()


def test_fused_per_tensor_fp8_quant_rejects_bad_fp8_dtype() -> None:
    x = torch.zeros((1, 1, 1), dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        fused_per_tensor_fp8_quant(x, fp8_dtype=torch.float32)


def test_fused_per_tensor_fp8_quant_rejects_non_3d() -> None:
    x = torch.zeros((4, 8), dtype=torch.bfloat16)
    with pytest.raises(AssertionError):
        fused_per_tensor_fp8_quant(x)


def test_fused_per_tensor_fp8_quant_rejects_non_contiguous_inner() -> None:
    x = torch.zeros((4, 8, 16), dtype=torch.bfloat16).transpose(1, 2)
    with pytest.raises(AssertionError):
        fused_per_tensor_fp8_quant(x)


def test_fused_per_tensor_fp8_quant_accepts_out_buffer() -> None:
    x = torch.randn(8, 4, 64, device=DEVICE, dtype=torch.bfloat16)
    out = torch.empty_like(x, dtype=FP8_DTYPE)
    outputs, _ = fused_per_tensor_fp8_quant(x, out=[out])
    assert outputs[0].data_ptr() == out.data_ptr()


def test_fused_per_tensor_fp8_quant_default_off_regression() -> None:
    """Default path must not depend on any opt-in env var and stay deterministic."""
    torch.manual_seed(0)
    x = torch.randn(32, 4, 128, device=DEVICE, dtype=torch.bfloat16)
    out_a, scale_a = fused_per_tensor_fp8_quant(x)
    out_b, scale_b = fused_per_tensor_fp8_quant(x)
    assert torch.equal(out_a[0].view(torch.int8), out_b[0].view(torch.int8))
    assert torch.equal(scale_a[0], scale_b[0])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
