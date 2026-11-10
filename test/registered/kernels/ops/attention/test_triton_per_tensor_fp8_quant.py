"""Lightweight tests for the fused per-tensor FP8 quantizer and the opt-in
FlyDSL FP8 MLA prefill capability gate.

These tests deliberately avoid importing the full ``sglang`` package: torch and
triton are optional (``importorskip``), and the modules under test are loaded by
file path. That keeps the suite collectable in a minimal CI environment where
``pip install -e python`` is not available.
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
    "_sglang_triton_per_tensor_fp8_quant_under_test",
)
_caps = _load_module(
    Path("python/sglang/kernels/ops/attention/mla_flydsl_fp8_caps.py"),
    "_sglang_mla_flydsl_fp8_caps_under_test",
)

_HAS_CUDA = torch.cuda.is_available()


# ---------------------------------------------------------------------------
# Capability probe / fallback (pure Python, no GPU required)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(_caps is None, reason="mla_flydsl_fp8_caps unavailable")
def test_default_off_never_selects_flydsl():
    assert not _caps.should_use_flydsl_fp8_prefill(
        enabled=False, gfx="gfx950", flydsl_available=True
    )


@pytest.mark.skipif(_caps is None, reason="mla_flydsl_fp8_caps unavailable")
def test_requires_gfx950():
    assert not _caps.should_use_flydsl_fp8_prefill(
        enabled=True, gfx="gfx942", flydsl_available=True
    )
    assert _caps.should_use_flydsl_fp8_prefill(
        enabled=True, gfx="gfx950", flydsl_available=True
    )


@pytest.mark.skipif(_caps is None, reason="mla_flydsl_fp8_caps unavailable")
def test_requires_flydsl():
    assert not _caps.should_use_flydsl_fp8_prefill(
        enabled=True, gfx="gfx950", flydsl_available=False
    )


@pytest.mark.skipif(_caps is None, reason="mla_flydsl_fp8_caps unavailable")
def test_unknown_gfx_falls_back():
    assert not _caps.should_use_flydsl_fp8_prefill(
        enabled=True, gfx=None, flydsl_available=True
    )


@pytest.mark.skipif(_caps is None, reason="mla_flydsl_fp8_caps unavailable")
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


# ---------------------------------------------------------------------------
# Fused per-tensor FP8 quantizer (GPU required)
# ---------------------------------------------------------------------------


def _reference_quant(x, fp8_dtype):
    amax = x.abs().float().max() if x.numel() else x.new_zeros((), dtype=torch.float)
    descale = (
        amax / torch.finfo(fp8_dtype).max
        if amax > 0
        else amax.new_tensor(1e-6)
    )
    x_fp8 = (x.float() / descale).clamp(
        -torch.finfo(fp8_dtype).max, torch.finfo(fp8_dtype).max
    ).to(fp8_dtype)
    return x_fp8, descale.reshape(1)


@pytest.mark.skipif(_quant is None, reason="triton_per_tensor_fp8_quant unavailable")
@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA/ROCm device required")
@pytest.mark.parametrize("num_tokens", [0, 1, 37, 8192])
@pytest.mark.parametrize("num_inputs", [1, 2, 3])
@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_correctness(num_tokens, num_inputs, fp8_dtype):
    torch.manual_seed(0)
    heads, dim = 12, 192
    inputs = [
        (torch.randn(num_tokens, heads, dim, device="cuda") * 4).to(torch.bfloat16)
        for _ in range(num_inputs)
    ]
    outputs, descales = _quant.fused_per_tensor_fp8_quant(
        *inputs, fp8_dtype=fp8_dtype
    )
    assert len(outputs) == num_inputs and len(descales) == num_inputs
    for x, x_fp8, descale in zip(inputs, outputs, descales):
        ref_fp8, ref_descale = _reference_quant(x, fp8_dtype)
        assert x_fp8.shape == x.shape and x_fp8.is_contiguous()
        torch.testing.assert_close(descale, ref_descale, rtol=1e-6, atol=0)
        ulp_diff = x_fp8.view(torch.int8).int() - ref_fp8.view(torch.int8).int()
        assert ulp_diff.numel() == 0 or ulp_diff.abs().max() <= 1


@pytest.mark.skipif(_quant is None, reason="triton_per_tensor_fp8_quant unavailable")
@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA/ROCm device required")
def test_empty_and_zero_inputs_get_min_descale():
    empty = torch.empty((0, 4, 8), dtype=torch.bfloat16, device="cuda")
    zeros = torch.zeros((16, 4, 8), dtype=torch.bfloat16, device="cuda")
    outputs, descales = _quant.fused_per_tensor_fp8_quant(empty, zeros)
    for descale in descales:
        assert descale.item() > 0
        assert descale.item() == pytest.approx(_quant._MIN_DESCALE)
    assert outputs[0].numel() == 0
    assert torch.count_nonzero(outputs[1].float()) == 0


@pytest.mark.skipif(_quant is None, reason="triton_per_tensor_fp8_quant unavailable")
@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA/ROCm device required")
def test_extreme_values_are_clamped():
    x = torch.full((8, 2, 4), 1e4, dtype=torch.bfloat16, device="cuda")
    outputs, descales = _quant.fused_per_tensor_fp8_quant(x)
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    assert outputs[0].float().abs().max().item() <= fp8_max
    assert descales[0].item() > 0


@pytest.mark.skipif(_quant is None, reason="triton_per_tensor_fp8_quant unavailable")
@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA/ROCm device required")
def test_strided_view_input():
    full = torch.randn((16, 8, 256), dtype=torch.bfloat16, device="cuda")
    v = full[..., 128:]
    assert v.stride(-1) == 1
    outputs, _ = _quant.fused_per_tensor_fp8_quant(v)
    ref_fp8, _ = _reference_quant(v, torch.float8_e4m3fn)
    ulp_diff = outputs[0].view(torch.int8).int() - ref_fp8.view(torch.int8).int()
    assert ulp_diff.abs().max() <= 1


@pytest.mark.skipif(_quant is None, reason="triton_per_tensor_fp8_quant unavailable")
@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA/ROCm device required")
def test_out_buffer_reuse():
    x = torch.randn((32, 4, 8), dtype=torch.bfloat16, device="cuda")
    out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    outputs, _ = _quant.fused_per_tensor_fp8_quant(x, out=[out])
    assert outputs[0].data_ptr() == out.data_ptr()


@pytest.mark.skipif(_quant is None, reason="triton_per_tensor_fp8_quant unavailable")
def test_rejects_bad_arity_and_dtype():
    x = torch.zeros((1, 1, 1), dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        _quant.fused_per_tensor_fp8_quant()
    with pytest.raises(ValueError):
        _quant.fused_per_tensor_fp8_quant(x, x, x, x)
    with pytest.raises(ValueError):
        _quant.fused_per_tensor_fp8_quant(x, fp8_dtype=torch.float32)
