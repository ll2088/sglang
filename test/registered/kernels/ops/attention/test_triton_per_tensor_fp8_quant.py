"""Lightweight tests for the fused per-tensor FP8 quantizer and the opt-in
FlyDSL FP8 MLA prefill capability gate.

These tests deliberately avoid importing the full ``sglang`` package so that CI
can collect and run them with a minimal dependency set (pytest + torch). The
modules under test are loaded by file path; when torch/triton or the target
files are unavailable the tests skip cleanly instead of failing collection.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="torch required for FP8 quant tests")


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    # test/registered/kernels/ops/attention/<this file> -> repo root
    return here.parents[5]


def _load_module(rel_path: str, name: str):
    target = _repo_root() / rel_path
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
    except Exception:  # noqa: BLE001 - optional deps (triton) may be missing
        sys.modules.pop(name, None)
        return None
    return module


_quant = _load_module(
    "python/sglang/kernels/ops/attention/triton_per_tensor_fp8_quant.py",
    "_sglang_triton_per_tensor_fp8_quant_under_test",
)
_caps = _load_module(
    "python/sglang/kernels/ops/attention/mla_flydsl_fp8_caps.py",
    "_sglang_mla_flydsl_fp8_caps_under_test",
)

_HAS_CUDA = torch.cuda.is_available()


def _reference_quant(x: torch.Tensor, fp8_dtype: torch.dtype):
    fp8_max = float(torch.finfo(fp8_dtype).max)
    amax = x.abs().float().max() if x.numel() else x.new_zeros((), dtype=torch.float)
    descale = amax / fp8_max if amax > 0 else amax.new_tensor(1e-6)
    x_fp8 = (x.float() / descale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    return x_fp8, descale.reshape(1)


@pytest.mark.skipif(_quant is None, reason="triton_per_tensor_fp8_quant unavailable")
@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA/ROCm device required")
@pytest.mark.parametrize("num_tokens", [0, 1, 37, 8192])
def test_fused_per_tensor_fp8_quant_matches_reference(num_tokens: int) -> None:
    torch.manual_seed(0)
    q = torch.randn(num_tokens, 12, 192, device="cuda") * 4
    k = torch.randn(num_tokens, 12, 192, device="cuda")
    kv = torch.randn(num_tokens, 12, 256, device="cuda")
    # V is a strided view of the kv_b_proj output, as in MLA prefill.
    v = kv[..., 128:]
    inputs = [x.to(torch.bfloat16) for x in (q, k, v)]
    outputs, descales = _quant.fused_per_tensor_fp8_quant(*inputs)
    for x, x_fp8, descale in zip(inputs, outputs, descales):
        ref_fp8, ref_descale = _reference_quant(x, torch.float8_e4m3fn)
        assert x_fp8.shape == x.shape and x_fp8.is_contiguous()
        torch.testing.assert_close(descale, ref_descale, rtol=1e-6, atol=0)
        # x * (1 / descale) may round one e4m3 step away from x / descale.
        ulp_diff = x_fp8.view(torch.int8).int() - ref_fp8.view(torch.int8).int()
        assert ulp_diff.numel() == 0 or ulp_diff.abs().max() <= 1


@pytest.mark.skipif(_quant is None, reason="triton_per_tensor_fp8_quant unavailable")
@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA/ROCm device required")
def test_fused_per_tensor_fp8_quant_empty_and_zero_inputs() -> None:
    empty = torch.empty(0, 4, 64, device="cuda", dtype=torch.bfloat16)
    zeros = torch.zeros(16, 4, 64, device="cuda", dtype=torch.bfloat16)
    outputs, descales = _quant.fused_per_tensor_fp8_quant(empty, zeros)
    assert outputs[0].numel() == 0
    assert outputs[1].shape == zeros.shape
    # Empty and all-zero inputs must get a positive descale to avoid NaNs.
    for d in descales:
        assert d.numel() == 1 and float(d) > 0
    assert torch.count_nonzero(outputs[1].float()) == 0


@pytest.mark.skipif(_quant is None, reason="triton_per_tensor_fp8_quant unavailable")
def test_fused_per_tensor_fp8_quant_rejects_bad_arity() -> None:
    with pytest.raises(ValueError):
        _quant.fused_per_tensor_fp8_quant()


@pytest.mark.skipif(_caps is None, reason="mla_flydsl_fp8_caps unavailable")
def test_default_off_never_selects_flydsl() -> None:
    assert not _caps.should_use_flydsl_fp8_prefill(
        enabled=False, gfx="gfx950", flydsl_available=True
    )


@pytest.mark.skipif(_caps is None, reason="mla_flydsl_fp8_caps unavailable")
def test_requires_gfx950_and_flydsl() -> None:
    assert not _caps.should_use_flydsl_fp8_prefill(
        enabled=True, gfx="gfx942", flydsl_available=True
    )
    assert not _caps.should_use_flydsl_fp8_prefill(
        enabled=True, gfx="gfx950", flydsl_available=False
    )
    assert not _caps.should_use_flydsl_fp8_prefill(
        enabled=True, gfx=None, flydsl_available=True
    )
    assert _caps.should_use_flydsl_fp8_prefill(
        enabled=True, gfx="gfx950", flydsl_available=True
    )


@pytest.mark.skipif(_caps is None, reason="mla_flydsl_fp8_caps unavailable")
def test_fallback_reason_is_explicit() -> None:
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
