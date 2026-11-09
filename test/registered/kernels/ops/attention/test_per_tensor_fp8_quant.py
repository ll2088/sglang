"""Tests for fused per-tensor FP8 quantization and the opt-in MLA prefill
backend selector. Skips cleanly when torch/triton or a GPU is unavailable."""

import os
import sys

import pytest

# The verification commands run from the repo root without `pip install -e python`.
# Make the in-tree package importable so the tests exercise the real modules.
_PY = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "python")
_PY = os.path.abspath(_PY)
if os.path.isdir(_PY) and _PY not in sys.path:
    sys.path.insert(0, _PY)

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm device required", allow_module_level=True)

from sglang.kernels.ops.attention._per_tensor_fp8_quant_triton import (  # noqa: E402
    _BLOCK,
    _PARTS,
    fused_per_tensor_fp8_quant,
    reference_per_tensor_fp8_quant,
)
from sglang.kernels.ops.attention.mla_prefill_fp8_selector import (  # noqa: E402
    BACKEND_NAME,
    resolve_mla_prefill_kernel,
    select_mla_prefill_backend,
)

FP8_DTYPE = torch.float8_e4m3fn
_STRIPE = _PARTS * _BLOCK


@pytest.mark.parametrize("numel", [0, 1, 37, 8192])
def test_fused_per_tensor_fp8_quant_matches_reference(numel):
    torch.manual_seed(0)
    x = (torch.randn(numel, device="cuda") * 4).to(torch.bfloat16)
    (out,), (descale,) = fused_per_tensor_fp8_quant(x)
    ref_fp8, ref_descale = reference_per_tensor_fp8_quant(x)
    assert out.shape == x.shape and out.is_contiguous()
    torch.testing.assert_close(descale, ref_descale, rtol=1e-6, atol=0)
    ulp = out.view(torch.int8).int() - ref_fp8.view(torch.int8).int()
    assert ulp.numel() == 0 or ulp.abs().max() <= 1


@pytest.mark.parametrize("numel", [1, _BLOCK, _BLOCK + 1, _STRIPE - 1, _STRIPE, _STRIPE + 1])
def test_fused_per_tensor_fp8_quant_covers_every_element(numel):
    x = torch.full((numel,), 0.5, device="cuda", dtype=torch.bfloat16)
    (out,), (descale,) = fused_per_tensor_fp8_quant(x)
    expected = 0.5 / (0.5 / 448.0)
    assert torch.allclose(out.float(), torch.full_like(out.float(), expected), atol=1.0)
    assert descale.item() > 0


def test_fused_per_tensor_fp8_quant_all_zero():
    x = torch.zeros(128, device="cuda", dtype=torch.bfloat16)
    (out,), (descale,) = fused_per_tensor_fp8_quant(x)
    assert torch.all(out.view(torch.int8) == 0)
    assert descale.item() > 0


def test_selector_default_off():
    assert select_mla_prefill_backend() is None
    assert select_mla_prefill_backend("ROCM_AITER_FA") is None
    assert resolve_mla_prefill_kernel() is None


def test_selector_unknown_backend_falls_back():
    # Unknown / unsupported names must not raise and must not select a kernel.
    assert select_mla_prefill_backend("NOT_A_BACKEND") is None
    assert resolve_mla_prefill_kernel("NOT_A_BACKEND") is None


def test_selector_opt_in_requires_support():
    # On non-gfx950 or without aiter, opt-in must fall back to the existing path.
    result = select_mla_prefill_backend(BACKEND_NAME)
    if result is None:
        assert resolve_mla_prefill_kernel() is None
    else:
        assert result == BACKEND_NAME


def test_import_smoke():
    import sglang.kernels.ops.attention as m

    assert hasattr(m, "register_kernel")


def test_import_smoke_without_opt_in():
    # Importing the attention package and resolving the default selector must not
    # require the optional FlyDSL/AITER stack.
    import sglang.kernels.ops.attention as m  # noqa: F401
    from sglang.kernels.ops.attention.mla_prefill_fp8_selector import (
        resolve_mla_prefill_kernel,
    )

    assert resolve_mla_prefill_kernel() is None


@pytest.mark.parametrize("numel", [0, 1, 37, 8192])
def test_perf_quant_median_latency(numel):
    """Performance dimension: record median latency of the fused quantizer.

    This is a smoke-level measurement (no baseline kernel in-tree yet); it fails
    only if the fused path is pathologically slow, and always prints the median
    so CI logs carry a number for the acceptance record.
    """
    import statistics
    import time

    x = (torch.randn(numel, device="cuda") * 4).to(torch.bfloat16)
    # warmup
    for _ in range(5):
        fused_per_tensor_fp8_quant(x)
    torch.cuda.synchronize()
    samples = []
    for _ in range(20):
        t0 = time.perf_counter()
        fused_per_tensor_fp8_quant(x)
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - t0)
    median = statistics.median(samples)
    print(f"perf numel={numel} median_ms={median * 1e3:.4f}")
    assert median < 1.0, f"fused quantizer unexpectedly slow: {median:.3f}s"
