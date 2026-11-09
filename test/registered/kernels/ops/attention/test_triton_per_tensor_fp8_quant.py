"""Standalone tests for the fused per-tensor FP8 quantization operator.

This module intentionally avoids importing the full ``sglang`` package so it
can run in a minimal environment (torch + triton + pytest only). The operator
module is loaded directly from its file path, which keeps the test entry point
independent of the package installation and of any optional runtime deps.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


def _load_quant_module():
    """Load ``triton_per_tensor_fp8_quant`` by file path, bypassing the package."""
    here = Path(__file__).resolve()
    # test/registered/kernels/ops/attention -> repo root
    repo_root = here.parents[5]
    mod_path = (
        repo_root
        / "python"
        / "sglang"
        / "kernels"
        / "ops"
        / "attention"
        / "triton_per_tensor_fp8_quant.py"
    )
    if not mod_path.is_file():
        return None
    spec = importlib.util.spec_from_file_location(
        "_sglang_triton_per_tensor_fp8_quant", mod_path
    )
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_quant_mod = _load_quant_module()

if _quant_mod is None:
    pytest.skip(
        "triton_per_tensor_fp8_quant module not found", allow_module_level=True
    )

fused_per_tensor_fp8_quant = _quant_mod.fused_per_tensor_fp8_quant
_BLOCK = _quant_mod._BLOCK
_PARTS = _quant_mod._PARTS
_MIN_DESCALE = _quant_mod._MIN_DESCALE

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = float(torch.finfo(FP8_DTYPE).max)


def _require_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm device required")


def _reference_quant(x: torch.Tensor):
    amax = x.abs().float().max() if x.numel() else x.new_zeros((), dtype=torch.float)
    descale = amax / FP8_MAX if amax > 0 else amax.new_tensor(_MIN_DESCALE)
    x_fp8 = (x.float() / descale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return x_fp8, descale.reshape(1)


def test_import_and_public_api():
    """The operator module imports and exposes the documented entry point."""
    assert callable(fused_per_tensor_fp8_quant)
    assert _BLOCK > 0 and _PARTS > 0
    assert _MIN_DESCALE > 0


@pytest.mark.parametrize("num_tokens", [0, 1, 37, 8192])
@pytest.mark.parametrize("num_inputs", [1, 2, 3])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_quant_matches_reference(num_tokens, num_inputs, dtype, fp8_dtype):
    _require_device()
    torch.manual_seed(0)
    heads, dim = 4, 128
    inputs = [
        (torch.randn(num_tokens, heads, dim, device="cuda") * (i + 1)).to(dtype)
        for i in range(num_inputs)
    ]
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
        ulp_diff = x_fp8.view(torch.int8).int() - ref_fp8.view(torch.int8).int()
        assert ulp_diff.numel() == 0 or ulp_diff.abs().max() <= 1


def test_empty_and_all_zero_inputs_get_min_descale():
    _require_device()
    empty = torch.empty((0, 2, 64), dtype=torch.bfloat16, device="cuda")
    zeros = torch.zeros((16, 2, 64), dtype=torch.bfloat16, device="cuda")
    outputs, descales = fused_per_tensor_fp8_quant(empty, zeros)
    for out, descale in zip(outputs, descales):
        assert torch.isfinite(descale).all()
        assert descale.item() == pytest.approx(_MIN_DESCALE)
        assert torch.isfinite(out.float()).all()


def test_extreme_values_are_clamped():
    _require_device()
    x = torch.full((8, 2, 64), 1e4, dtype=torch.bfloat16, device="cuda")
    y = torch.full((8, 2, 64), -1e4, dtype=torch.bfloat16, device="cuda")
    outputs, descales = fused_per_tensor_fp8_quant(x, y)
    for out, descale in zip(outputs, descales):
        assert torch.isfinite(descale).all() and descale.item() > 0
        assert torch.isfinite(out.float()).all()
        assert out.float().abs().max().item() <= FP8_MAX + 1e-3


_STRIPE = _PARTS * _BLOCK


@pytest.mark.parametrize(
    "numel",
    [1, _BLOCK, _BLOCK + 1, _STRIPE - 1, _STRIPE, _STRIPE + 1, 3 * _STRIPE + 7],
)
def test_covers_every_element_across_stripes(numel):
    """One grid serves inputs of different sizes; no element may be dropped."""
    _require_device()
    x = torch.full((numel, 1, 1), 0.5, dtype=torch.bfloat16, device="cuda")
    expected = 100.0 / FP8_MAX
    for pos in sorted({0, _BLOCK - 1, _BLOCK, _STRIPE - 1, _STRIPE, numel - 1}):
        if pos >= numel:
            continue
        x[pos] = 100.0
        outputs, descales = fused_per_tensor_fp8_quant(x)
        assert descales[0].item() == pytest.approx(expected, rel=1e-6)
        assert outputs[0][pos].float().item() == pytest.approx(FP8_MAX, rel=1e-2)
        x[pos] = 0.5


def test_strided_view_input():
    _require_device()
    s, h, nope, v_head = 16, 4, 128, 128
    kv = torch.randn((s, h, nope + v_head), dtype=torch.bfloat16, device="cuda")
    v = kv[..., nope:]
    assert v.stride(-1) == 1
    outputs, descales = fused_per_tensor_fp8_quant(v)
    ref_fp8, ref_descale = _reference_quant(v)
    torch.testing.assert_close(descales[0], ref_descale, rtol=1e-6, atol=0)
    ulp_diff = outputs[0].view(torch.int8).int() - ref_fp8.view(torch.int8).int()
    assert ulp_diff.abs().max() <= 1


def test_out_buffers_are_reused():
    _require_device()
    x = torch.randn((32, 2, 64), dtype=torch.bfloat16, device="cuda")
    out = torch.empty(x.shape, dtype=FP8_DTYPE, device="cuda")
    outputs, _ = fused_per_tensor_fp8_quant(x, out=[out])
    assert outputs[0].data_ptr() == out.data_ptr()


def test_rejects_bad_inputs():
    _require_device()
    x = torch.randn((4, 2, 64), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError):
        fused_per_tensor_fp8_quant()
    with pytest.raises(ValueError):
        fused_per_tensor_fp8_quant(x, x, x, x)
    with pytest.raises(ValueError):
        fused_per_tensor_fp8_quant(x, fp8_dtype=torch.float32)
    with pytest.raises(AssertionError):
        fused_per_tensor_fp8_quant(x.transpose(1, 2))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
