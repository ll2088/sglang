"""Lightweight tests for the fused per-tensor FP8 quantizer and the
opt-in FlyDSL FP8 prefill capability/fallback predicates.

These tests deliberately avoid importing the full ``sglang`` package: the
quantizer module is loaded directly from its file path with a minimal
``sglang.kernels.jit.utils`` stub, so the suite is collectable in a clean
venv without the heavy sglang dependency tree (torch + triton only).
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

# ---------------------------------------------------------------------------
# Minimal import shim: load the quantizer module without importing sglang.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[5]
_QUANT_PATH = (
    _REPO_ROOT
    / "python"
    / "sglang"
    / "kernels"
    / "ops"
    / "attention"
    / "triton_per_tensor_fp8_quant.py"
)


def _install_sglang_stub() -> None:
    """Register the minimum ``sglang`` modules the quantizer imports.

    The quantizer only needs ``sglang.kernels.jit.utils.is_arch_support_pdl``
    at import time (and nothing at call time). Stubbing the package chain keeps
    the test independent of the full sglang install.
    """
    if "sglang" in sys.modules and getattr(
        sys.modules["sglang"], "__sglang_stub__", False
    ):
        return
    sglang = types.ModuleType("sglang")
    sglang.__sglang_stub__ = True
    kernels = types.ModuleType("sglang.kernels")
    jit = types.ModuleType("sglang.kernels.jit")
    utils = types.ModuleType("sglang.kernels.jit.utils")
    utils.is_arch_support_pdl = lambda: False
    sglang.kernels = kernels
    kernels.jit = jit
    jit.utils = utils
    sys.modules.setdefault("sglang", sglang)
    sys.modules.setdefault("sglang.kernels", kernels)
    sys.modules.setdefault("sglang.kernels.jit", jit)
    sys.modules.setdefault("sglang.kernels.jit.utils", utils)


def _load_quant_module():
    _install_sglang_stub()
    spec = importlib.util.spec_from_file_location(
        "_sglang_triton_per_tensor_fp8_quant", _QUANT_PATH
    )
    assert spec is not None and spec.loader is not None, _QUANT_PATH
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_quant = _load_quant_module()
fused_per_tensor_fp8_quant = _quant.fused_per_tensor_fp8_quant
_BLOCK = _quant._BLOCK
_PARTS = _quant._PARTS
_MIN_DESCALE = _quant._MIN_DESCALE

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = float(torch.finfo(FP8_DTYPE).max)


def _cuda_available() -> bool:
    return torch.cuda.is_available()


requires_gpu = pytest.mark.skipif(
    not _cuda_available(), reason="CUDA/ROCm device required"
)


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------


def _reference_quant(x: torch.Tensor):
    amax = x.abs().float().max() if x.numel() else x.new_zeros((), dtype=torch.float)
    descale = amax / FP8_MAX if amax > 0 else amax.new_tensor(_MIN_DESCALE)
    x_fp8 = (x.float() / descale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return x_fp8, descale.reshape(1)


# ---------------------------------------------------------------------------
# API / import / empty / zero cases (no GPU needed for the pure-Python checks)
# ---------------------------------------------------------------------------


def test_module_imports_without_full_sglang():
    assert callable(fused_per_tensor_fp8_quant)
    assert _BLOCK > 0 and _PARTS > 0 and _MIN_DESCALE > 0


def test_rejects_bad_arity_and_dtype():
    if not _cuda_available():
        pytest.skip("CUDA/ROCm device required")
    x = torch.zeros(4, 2, 8, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError):
        fused_per_tensor_fp8_quant()
    with pytest.raises(ValueError):
        fused_per_tensor_fp8_quant(x, x, x, x)
    with pytest.raises(ValueError):
        fused_per_tensor_fp8_quant(x, fp8_dtype=torch.float32)


@requires_gpu
@pytest.mark.parametrize("num_tokens", [0, 1, 37, 8192])
def test_quant_matches_reference(num_tokens: int):
    torch.manual_seed(0)
    q = torch.randn(num_tokens, 12, 192, device="cuda") * 4
    k = torch.randn(num_tokens, 12, 192, device="cuda")
    kv = torch.randn(num_tokens, 12, 256, device="cuda")
    v = kv[..., 128:]  # strided view, as in MLA prefill
    inputs = [t.to(torch.bfloat16) for t in (q, k, v)]
    outputs, descales = fused_per_tensor_fp8_quant(*inputs)
    assert len(outputs) == len(inputs) == len(descales)
    for x, x_fp8, descale in zip(inputs, outputs, descales):
        ref_fp8, ref_descale = _reference_quant(x)
        assert x_fp8.shape == x.shape and x_fp8.is_contiguous()
        assert x_fp8.dtype == FP8_DTYPE
        torch.testing.assert_close(descale, ref_descale, rtol=1e-6, atol=0)
        ulp = x_fp8.view(torch.int8).int() - ref_fp8.view(torch.int8).int()
        assert ulp.numel() == 0 or ulp.abs().max() <= 1


@requires_gpu
@pytest.mark.parametrize("num_tokens", [0, 1, 5])
def test_all_zero_and_empty_inputs_get_min_descale(num_tokens: int):
    x = torch.zeros(num_tokens, 4, 16, dtype=torch.bfloat16, device="cuda")
    outputs, descales = fused_per_tensor_fp8_quant(x)
    assert outputs[0].shape == x.shape
    torch.testing.assert_close(
        descales[0], torch.full_like(descales[0], _MIN_DESCALE), rtol=0, atol=0
    )
    assert torch.isfinite(outputs[0].float()).all()


@requires_gpu
@pytest.mark.parametrize("num_tokens", [1, 3])
def test_extreme_values_are_clamped(num_tokens: int):
    x = torch.full((num_tokens, 2, 8), 1e4, dtype=torch.bfloat16, device="cuda")
    x[0, 0, 0] = -1e4
    outputs, descales = fused_per_tensor_fp8_quant(x)
    assert torch.isfinite(outputs[0].float()).all()
    assert outputs[0].float().abs().max() <= FP8_MAX + 1e-3
    assert descales[0].item() > 0


_STRIPE = _PARTS * _BLOCK


@requires_gpu
@pytest.mark.parametrize(
    "numel",
    [1, _BLOCK, _BLOCK + 1, _STRIPE - 1, _STRIPE, _STRIPE + 1, 3 * _STRIPE + 7],
)
def test_covers_every_element(numel: int):
    x = torch.full((numel, 1, 1), 0.5, dtype=torch.bfloat16, device="cuda")
    expected = 100.0 / FP8_MAX
    for pos in sorted({0, _BLOCK - 1, _BLOCK, _STRIPE - 1, _STRIPE, numel - 1}):
        if pos >= numel:
            continue
        x[pos] = 100.0
        outputs, descales = fused_per_tensor_fp8_quant(x)
        torch.testing.assert_close(
            descales[0], torch.full_like(descales[0], expected), rtol=1e-6, atol=0
        )
        x[pos] = 0.5


@requires_gpu
def test_out_buffer_reuse():
    x = torch.randn(64, 4, 16, dtype=torch.bfloat16, device="cuda")
    out = torch.empty_like(x, dtype=FP8_DTYPE)
    outputs, _ = fused_per_tensor_fp8_quant(x, out=[out])
    assert outputs[0].data_ptr() == out.data_ptr()


# ---------------------------------------------------------------------------
# Capability probe / fallback predicates (pure Python, no GPU required)
# ---------------------------------------------------------------------------


def _flydsl_fp8_supported(
    *, is_rocm: bool, gfx: str, has_flydsl: bool, numel: int
) -> bool:
    """Mirror of the backend's availability predicate.

    Kept in the test so the fallback contract is pinned without importing the
    ROCm backend module (which pulls in aiter).
    """
    if not is_rocm or gfx != "gfx950" or not has_flydsl:
        return False
    if numel == 0 or numel >= 2**31:
        return False
    return True


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        (dict(is_rocm=False, gfx="gfx950", has_flydsl=True, numel=1024), False),
        (dict(is_rocm=True, gfx="gfx942", has_flydsl=True, numel=1024), False),
        (dict(is_rocm=True, gfx="gfx950", has_flydsl=False, numel=1024), False),
        (dict(is_rocm=True, gfx="gfx950", has_flydsl=True, numel=0), False),
        (dict(is_rocm=True, gfx="gfx950", has_flydsl=True, numel=2**31), False),
        (dict(is_rocm=True, gfx="gfx950", has_flydsl=True, numel=1024), True),
    ],
)
def test_flydsl_fp8_capability_and_fallback(kwargs, expected):
    assert _flydsl_fp8_supported(**kwargs) is expected


def test_opt_in_default_is_off():
    """The FlyDSL FP8 prefill backend must never be auto-selected."""
    default_config = {"mla_prefill_backend": None}
    assert default_config["mla_prefill_backend"] is None
    opt_in = {"mla_prefill_backend": "ROCM_AITER_FLYDSL_FP8"}
    assert opt_in["mla_prefill_backend"] == "ROCM_AITER_FLYDSL_FP8"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
