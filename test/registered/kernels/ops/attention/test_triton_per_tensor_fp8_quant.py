"""Lightweight tests for the fused per-tensor FP8 quantizer.

Deliberately avoids importing the full ``sglang`` package so the job only
needs pytest + torch + triton. The quantizer module itself only depends on
``torch`` and ``triton``, so it can be imported directly by file path when the
package is not installed.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


def _load_quant_module():
    """Import the quantizer without importing the ``sglang`` package."""
    here = Path(__file__).resolve()
    # test/registered/kernels/ops/attention/ -> repo root
    root = here.parents[5]
    target = (
        root
        / "python"
        / "sglang"
        / "kernels"
        / "ops"
        / "attention"
        / "triton_per_tensor_fp8_quant.py"
    )
    if not target.is_file():
        return None
    spec = importlib.util.spec_from_file_location(
        "_sglang_triton_per_tensor_fp8_quant", target
    )
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_MOD = _load_quant_module()

if _MOD is None:
    pytest.skip(
        "triton_per_tensor_fp8_quant.py not found; skipping quantizer tests",
        allow_module_level=True,
    )

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm device required", allow_module_level=True)

fused_per_tensor_fp8_quant = _MOD.fused_per_tensor_fp8_quant
_BLOCK = _MOD._BLOCK
_PARTS = _MOD._PARTS
_MIN_DESCALE = _MOD._MIN_DESCALE

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = float(torch.finfo(FP8_DTYPE).max)
_STRIPE = _PARTS * _BLOCK


def _reference_quant(x: torch.Tensor):
    amax = x.abs().float().max() if x.numel() else x.new_zeros((), dtype=torch.float)
    descale = amax / FP8_MAX if amax > 0 else amax.new_tensor(_MIN_DESCALE)
    x_fp8 = (x.float() / descale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return x_fp8, descale.reshape(1)


@pytest.mark.parametrize("num_tokens", [0, 1, 37, 8192])
@pytest.mark.parametrize("num_inputs", [1, 2, 3])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_correctness(num_tokens, num_inputs, dtype, fp8_dtype):
    torch.manual_seed(0)
    heads, dim = 4, 128
    base = [
        (torch.randn(num_tokens, heads, dim, device="cuda") * 4).to(dtype),
        torch.randn(num_tokens, heads, dim, device="cuda").to(dtype),
        torch.randn(num_tokens, heads, dim, device="cuda").to(dtype),
    ]
    inputs = base[:num_inputs]
    outputs, descales = fused_per_tensor_fp8_quant(*inputs, fp8_dtype=fp8_dtype)
    assert len(outputs) == num_inputs and len(descales) == num_inputs
    fp8_max = float(torch.finfo(fp8_dtype).max)
    for x, x_fp8, descale in zip(inputs, outputs, descales):
        assert x_fp8.shape == x.shape and x_fp8.is_contiguous()
        assert x_fp8.dtype == fp8_dtype
        assert descale.numel() == 1 and descale.dtype == torch.float32
        amax = x.abs().float().max() if x.numel() else x.new_zeros((), dtype=torch.float)
        ref_descale = amax / fp8_max if amax > 0 else amax.new_tensor(_MIN_DESCALE)
        torch.testing.assert_close(descale, ref_descale.reshape(1), rtol=1e-6, atol=0)
        ref_fp8 = (x.float() / ref_descale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
        ulp = x_fp8.view(torch.int8).int() - ref_fp8.view(torch.int8).int()
        assert ulp.numel() == 0 or ulp.abs().max() <= 1


def test_empty_and_zero_inputs_get_min_descale():
    empty = torch.empty(0, 2, 8, device="cuda", dtype=torch.bfloat16)
    zeros = torch.zeros(16, 2, 8, device="cuda", dtype=torch.bfloat16)
    outputs, descales = fused_per_tensor_fp8_quant(empty, zeros)
    for descale in descales:
        torch.testing.assert_close(
            descale, torch.full_like(descale, _MIN_DESCALE), rtol=0, atol=0
        )
    assert outputs[0].numel() == 0
    assert torch.count_nonzero(outputs[1].float()) == 0


def test_extreme_values_clamped():
    x = torch.full((8, 2, 8), 1e4, device="cuda", dtype=torch.bfloat16)
    (out,), (descale,) = fused_per_tensor_fp8_quant(x)
    assert torch.isfinite(out.float()).all()
    assert out.float().abs().max() <= FP8_MAX
    assert descale.item() > 0


@pytest.mark.parametrize(
    "numel",
    [1, _BLOCK, _BLOCK + 1, _STRIPE - 1, _STRIPE, _STRIPE + 1, 3 * _STRIPE + 7],
)
def test_covers_every_element(numel):
    """One grid serves inputs of different sizes; no element may be dropped."""
    x = torch.full((numel, 1, 1), 0.5, device="cuda", dtype=torch.bfloat16)
    (out,), (descale,) = fused_per_tensor_fp8_quant(x)
    expected = 0.5 / descale.item()
    got = out.float().flatten()
    assert got.numel() == numel
    torch.testing.assert_close(
        got, torch.full_like(got, expected), rtol=1e-2, atol=1e-2
    )


def test_strided_view_input():
    s, h, nope, v_head = 16, 4, 128, 128
    kv = torch.randn(s, h, nope + v_head, device="cuda", dtype=torch.bfloat16)
    v = kv[..., nope:]
    assert v.stride(-1) == 1
    (out,), (descale,) = fused_per_tensor_fp8_quant(v)
    ref_fp8, ref_descale = _reference_quant(v)
    torch.testing.assert_close(descale, ref_descale, rtol=1e-6, atol=0)
    ulp = out.view(torch.int8).int() - ref_fp8.view(torch.int8).int()
    assert ulp.numel() == 0 or ulp.abs().max() <= 1


def test_rejects_bad_inputs():
    x = torch.randn(4, 2, 8, device="cuda", dtype=torch.bfloat16)
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
