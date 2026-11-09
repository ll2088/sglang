"""Tests for the fused per-tensor FP8 quantizer used by MLA prefill.

This module is intentionally importable without a full ``sglang`` install:
the kernel module is loaded by file path via ``importlib`` so that importing
``sglang`` (and its heavy dependency tree) is never triggered during test
collection. When ``torch``/``triton`` or a GPU are unavailable the tests are
skipped instead of failing.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

# Optional imports: keep this module collectable in a minimal environment where
# the full sglang package (and its CI helpers) is not installed.
try:
    import torch
except Exception:  # noqa: BLE001
    torch = None

try:
    import triton  # noqa: F401
except Exception:  # noqa: BLE001
    triton = None


def _load_quant_module():
    """Load ``triton_per_tensor_fp8_quant`` by file path, bypassing the
    ``sglang`` package ``__init__`` (which pulls in the full dependency set)."""
    here = Path(__file__).resolve()
    # test/registered/kernels/ops/attention/ -> repo root -> python/sglang/...
    repo_root = here.parents[5]
    target = (
        repo_root
        / "python"
        / "sglang"
        / "kernels"
        / "ops"
        / "attention"
        / "triton_per_tensor_fp8_quant.py"
    )
    if not target.is_file():
        return None
    name = "_sglang_triton_per_tensor_fp8_quant_under_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, target)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:  # noqa: BLE001 - missing triton/torch at import time
        sys.modules.pop(name, None)
        return None
    return module


_quant = _load_quant_module()

pytestmark = pytest.mark.skipif(
    _quant is None or torch is None or not torch.cuda.is_available(),
    reason="fused per-tensor FP8 quantizer or CUDA/ROCm device unavailable",
)

FP8_DTYPE = None
if torch is not None:
    FP8_DTYPE = torch.float8_e4m3fn


def _reference_quant(x):
    amax = x.abs().float().max() if x.numel() else x.new_zeros((), dtype=torch.float)
    descale = (
        amax / torch.finfo(FP8_DTYPE).max if amax > 0 else amax.new_tensor(1e-6)
    )
    x_fp8 = (x.float() / descale).clamp(-448, 448).to(FP8_DTYPE)
    return x_fp8, descale.reshape(1)


@pytest.mark.parametrize("num_tokens", [0, 1, 37, 8192])
def test_fused_per_tensor_fp8_quant_matches_reference(num_tokens):
    torch.manual_seed(0)
    q = torch.randn(num_tokens, 12, 192, device="cuda") * 4
    k = torch.randn(num_tokens, 12, 192, device="cuda")
    kv = torch.randn(num_tokens, 12, 256, device="cuda")
    # V is a strided view of the kv_b_proj output, as in MLA prefill.
    v = kv[..., 128:]
    inputs = [x.to(torch.bfloat16) for x in (q, k, v)]
    outputs, descales = _quant.fused_per_tensor_fp8_quant(*inputs)
    for x, x_fp8, descale in zip(inputs, outputs, descales):
        ref_fp8, ref_descale = _reference_quant(x)
        assert x_fp8.shape == x.shape and x_fp8.is_contiguous()
        torch.testing.assert_close(descale, ref_descale, rtol=1e-6, atol=0)
        # x * (1 / descale) may round one e4m3 step away from x / descale.
        ulp_diff = x_fp8.view(torch.int8).int() - ref_fp8.view(torch.int8).int()
        assert ulp_diff.numel() == 0 or ulp_diff.abs().max() <= 1


_STRIPE = _quant._PARTS * _quant._BLOCK if _quant is not None else 0


@pytest.mark.parametrize(
    "numel",
    [1, 4096, 4097, _STRIPE - 1, _STRIPE, _STRIPE + 1, 3 * _STRIPE + 7],
)
def test_fused_per_tensor_fp8_quant_covers_every_element(numel):
    """One grid serves inputs of different sizes; no element may be dropped."""
    x = torch.full((numel, 1, 1), 0.5, device="cuda", dtype=torch.bfloat16)
    expected = 100.0 / torch.finfo(FP8_DTYPE).max
    for pos in sorted({0, 4095, 4096, _STRIPE - 1, _STRIPE, numel - 1}):
        if pos >= numel:
            continue
        y = x.clone()
        y.view(-1)[pos] = 100.0
        (out,), (descale,) = _quant.fused_per_tensor_fp8_quant(y)
        torch.testing.assert_close(descale, y.new_tensor(expected).reshape(1))
        assert out.view(-1)[pos].float().item() == pytest.approx(
            torch.finfo(FP8_DTYPE).max, rel=1e-2
        )


def test_fused_per_tensor_fp8_quant_empty_and_zero_inputs():
    """Empty and all-zero inputs get a positive descale (no NaN downstream)."""
    empty = torch.empty((0, 4, 8), device="cuda", dtype=torch.bfloat16)
    zeros = torch.zeros((16, 4, 8), device="cuda", dtype=torch.bfloat16)
    (e_out, z_out), (e_desc, z_desc) = _quant.fused_per_tensor_fp8_quant(
        empty, zeros
    )
    assert e_out.shape == empty.shape and z_out.shape == zeros.shape
    assert e_desc.item() > 0 and z_desc.item() > 0
    assert torch.isfinite(z_out.float()).all()


def test_fused_per_tensor_fp8_quant_rejects_bad_inputs():
    x = torch.zeros((4, 2, 8), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        _quant.fused_per_tensor_fp8_quant()
    with pytest.raises(ValueError):
        _quant.fused_per_tensor_fp8_quant(x, x, x, x)
    with pytest.raises(ValueError):
        _quant.fused_per_tensor_fp8_quant(x, fp8_dtype=torch.float32)
