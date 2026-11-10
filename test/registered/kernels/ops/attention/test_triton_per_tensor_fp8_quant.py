"""Dependency-light tests for the fused per-tensor FP8 quantizer and the
opt-in FlyDSL FP8 MLA prefill capability predicate.

These tests deliberately avoid importing the full ``sglang`` package: the
modules under test are loaded by file path so that a minimal CI environment
(pytest + torch + triton) can collect and run them. GPU-only cases skip
cleanly when no CUDA/ROCm device or triton is available.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="torch required for FP8 quant tests")


def _load_module(rel_path, name):
    here = Path(__file__).resolve()
    repo_root = here.parents[5]
    target = repo_root / rel_path
    if not target.is_file():
        return None
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, target)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:  # noqa: BLE001
        sys.modules.pop(name, None)
        return None
    return module


_quant = _load_module(
    Path("python/sglang/kernels/ops/attention/triton_per_tensor_fp8_quant.py"),
    "_sglang_per_tensor_fp8_quant_under_test",
)
_caps = _load_module(
    Path("python/sglang/kernels/ops/attention/mla_flydsl_fp8_caps.py"),
    "_sglang_mla_flydsl_fp8_caps_under_test",
)

_HAS_CUDA = torch.cuda.is_available()


def _require_quant():
    if _quant is None:
        pytest.skip("triton_per_tensor_fp8_quant unavailable")
    if not _HAS_CUDA:
        pytest.skip("CUDA/ROCm device required")


def _reference_quant(x, fp8_dtype):
    fp8_max = float(torch.finfo(fp8_dtype).max)
    amax = x.abs().float().max() if x.numel() else x.new_zeros((), dtype=torch.float)
    descale = amax / fp8_max if amax > 0 else amax.new_tensor(1e-6)
    x_fp8 = (x.float() / descale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    return x_fp8, descale.reshape(1)


@pytest.mark.parametrize("num_tokens", [0, 1, 37, 4096, 8192])
@pytest.mark.parametrize("num_inputs", [1, 2, 3])
def test_correctness(num_tokens, num_inputs):
    _require_quant()
    torch.manual_seed(0)
    heads, dim = 4, 128
    inputs = [
        (torch.randn(num_tokens, heads, dim, device="cuda") * (i + 1)).to(
            torch.bfloat16
        )
        for i in range(num_inputs)
    ]
    outputs, descales = _quant.fused_per_tensor_fp8_quant(*inputs)
    assert len(outputs) == num_inputs and len(descales) == num_inputs
    for x, x_fp8, descale in zip(inputs, outputs, descales):
        ref_fp8, ref_descale = _reference_quant(x, torch.float8_e4m3fn)
        assert x_fp8.shape == x.shape and x_fp8.is_contiguous()
        torch.testing.assert_close(descale, ref_descale, rtol=1e-6, atol=0)
        ulp = x_fp8.view(torch.int8).int() - ref_fp8.view(torch.int8).int()
        assert ulp.numel() == 0 or ulp.abs().max() <= 1


def test_empty_and_zero_inputs_get_positive_descale():
    _require_quant()
    empty = torch.empty(0, 2, 64, dtype=torch.bfloat16, device="cuda")
    zeros = torch.zeros(8, 2, 64, dtype=torch.bfloat16, device="cuda")
    outputs, descales = _quant.fused_per_tensor_fp8_quant(empty, zeros)
    for out, descale in zip(outputs, descales):
        assert descale.numel() == 1 and float(descale) > 0
        assert torch.isfinite(out.float()).all()


def test_strided_view_input():
    _require_quant()
    full = torch.randn(16, 4, 256, dtype=torch.bfloat16, device="cuda")
    view = full[..., 128:]
    assert view.stride(-1) == 1
    outputs, _ = _quant.fused_per_tensor_fp8_quant(view)
    ref_fp8, _ = _reference_quant(view, torch.float8_e4m3fn)
    ulp = outputs[0].view(torch.int8).int() - ref_fp8.view(torch.int8).int()
    assert ulp.abs().max() <= 1


def test_rejects_bad_arity():
    if _quant is None:
        pytest.skip("triton_per_tensor_fp8_quant unavailable")
    with pytest.raises(ValueError):
        _quant.fused_per_tensor_fp8_quant()


pytestmark_caps = pytest.mark.skipif(
    _caps is None, reason="mla_flydsl_fp8_caps unavailable"
)


@pytestmark_caps
def test_default_off_never_selects_flydsl():
    assert not _caps.should_use_flydsl_fp8_prefill(
        enabled=False, gfx="gfx950", flydsl_available=True
    )


@pytestmark_caps
def test_requires_gfx950():
    assert not _caps.should_use_flydsl_fp8_prefill(
        enabled=True, gfx="gfx942", flydsl_available=True
    )
    assert _caps.should_use_flydsl_fp8_prefill(
        enabled=True, gfx="gfx950", flydsl_available=True
    )


@pytestmark_caps
def test_requires_flydsl():
    assert not _caps.should_use_flydsl_fp8_prefill(
        enabled=True, gfx="gfx950", flydsl_available=False
    )


@pytestmark_caps
def test_unknown_gfx_falls_back():
    assert not _caps.should_use_flydsl_fp8_prefill(
        enabled=True, gfx=None, flydsl_available=True
    )


@pytestmark_caps
def test_fallback_reason_is_explicit():
    reason = _caps.flydsl_fp8_fallback_reason(
        enabled=True, gfx="gfx942", flydsl_available=True
    )
    assert reason is not None and "gfx950" in reason
    assert (
        _caps.flydsl_fp8_fallback_reason(
            enabled=True, gfx="gfx950", flydsl_available=True
        )
        is None
    )
