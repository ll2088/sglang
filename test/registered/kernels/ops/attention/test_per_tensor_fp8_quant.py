"""Tests for fused per-tensor FP8 quantization and its lazy import contract."""

import sys

import pytest
import torch

from sglang.kernels.ops.attention.per_tensor_fp8_quant import (
    _reference_quant,
    fused_per_tensor_fp8_quant,
)

FP8_DTYPE = torch.float8_e4m3fn


def test_import_without_triton(monkeypatch):
    """Importing the module and registry must not require triton."""
    monkeypatch.setitem(sys.modules, "triton", None)
    import importlib

    import sglang.kernels.ops.attention.per_tensor_fp8_quant as mod

    importlib.reload(mod)
    assert hasattr(mod, "fused_per_tensor_fp8_quant")


def test_reference_quant_matches_expected():
    x = torch.tensor([[[1.0, -2.0]]], dtype=torch.bfloat16)
    x_fp8, descale = _reference_quant(x)
    assert x_fp8.dtype == FP8_DTYPE
    assert descale.numel() == 1 and descale.item() > 0
    torch.testing.assert_close(
        x_fp8.float(), (x.float() / descale).clamp(-448, 448), rtol=0, atol=0
    )


@pytest.mark.parametrize("numel", [0, 1, 37, 8192])
@pytest.mark.parametrize("num_inputs", [1, 2, 3])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.skipif(
    __import__("importlib").util.find_spec("triton") is None, reason="needs triton"
)
def test_fused_quant_matches_reference(numel: int, num_inputs: int):
    torch.manual_seed(0)
    inputs = [
        torch.randn(numel, 4, 128, device="cuda", dtype=torch.bfloat16) * (i + 1)
        for i in range(num_inputs)
    ]
    outs, descales = fused_per_tensor_fp8_quant(*inputs)
    assert len(outs) == len(descales) == num_inputs
    for x, out, descale in zip(inputs, outs, descales):
        ref_fp8, ref_descale = _reference_quant(x)
        assert out.shape == x.shape and out.is_contiguous()
        torch.testing.assert_close(descale, ref_descale, rtol=1e-6, atol=0)
        ulp = out.view(torch.int8).int() - ref_fp8.view(torch.int8).int()
        assert ulp.numel() == 0 or ulp.abs().max() <= 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.skipif(
    __import__("importlib").util.find_spec("triton") is None, reason="needs triton"
)
def test_fused_quant_all_zero_and_extreme():
    x = torch.zeros(16, 4, 128, device="cuda", dtype=torch.bfloat16)
    outs, descales = fused_per_tensor_fp8_quant(x)
    assert torch.isfinite(outs[0].float()).all()
    assert descales[0].item() > 0
    x.fill_(1e4)
    outs, _ = fused_per_tensor_fp8_quant(x)
    assert outs[0].float().abs().max() <= 448


def test_rejects_bad_input_count():
    with pytest.raises(ValueError):
        fused_per_tensor_fp8_quant()
