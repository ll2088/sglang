"""Numerical tests for the fused per-tensor FP8 quantizer.

Deliberately lightweight: the module under test is loaded by file path with
``importlib`` so this test does not require a full ``sglang`` install (and its
heavy dependency tree) to run. When ``torch``/``triton`` or a GPU are missing
the whole module is skipped instead of failing at import time.
"""

import importlib.util
import sys
from pathlib import Path

import pytest


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


torch = pytest.importorskip("torch")
pytest.importorskip("triton")

_quant = _load_module(
    Path("python/sglang/kernels/ops/attention/triton_per_tensor_fp8_quant.py"),
    "_sglang_triton_per_tensor_fp8_quant_under_test",
)

pytestmark = [
    pytest.mark.skipif(_quant is None, reason="triton_per_tensor_fp8_quant unavailable"),
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA/ROCm device required"
    ),
]

FP8_DTYPE = torch.float8_e4m3fn


def _reference_quant(x):
    amax = x.abs().float().max() if x.numel() else x.new_zeros((), dtype=torch.float)
    descale = (
        amax / torch.finfo(FP8_DTYPE).max if amax > 0 else amax.new_tensor(1e-6)
    )
    x_fp8 = (x.float() / descale).clamp(-448, 448).to(FP8_DTYPE)
    return x_fp8, descale.reshape(1)


@pytest.mark.parametrize("num_tokens", [0, 1, 37, 8192])
def test_quant_matches_reference(num_tokens):
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


@pytest.mark.parametrize("num_tokens", [0, 1, 5])
def test_all_zero_and_empty_inputs_get_min_descale(num_tokens):
    x = torch.zeros(num_tokens, 4, 64, device="cuda", dtype=torch.bfloat16)
    outputs, descales = _quant.fused_per_tensor_fp8_quant(x)
    assert outputs[0].shape == x.shape
    assert descales[0].item() == pytest.approx(_quant._MIN_DESCALE)
    assert torch.isfinite(outputs[0].float()).all()


def test_single_input_and_out_buffer():
    x = torch.randn(64, 8, 128, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(x.shape, dtype=FP8_DTYPE, device="cuda")
    outputs, descales = _quant.fused_per_tensor_fp8_quant(x, out=[out])
    assert outputs[0] is out
    ref_fp8, ref_descale = _reference_quant(x)
    torch.testing.assert_close(descales[0], ref_descale, rtol=1e-6, atol=0)
    ulp_diff = out.view(torch.int8).int() - ref_fp8.view(torch.int8).int()
    assert ulp_diff.abs().max() <= 1


def test_rejects_bad_arity_and_dtype():
    x = torch.randn(4, 2, 8, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        _quant.fused_per_tensor_fp8_quant()
    with pytest.raises(ValueError):
        _quant.fused_per_tensor_fp8_quant(x, x, x, x)
    with pytest.raises(ValueError):
        _quant.fused_per_tensor_fp8_quant(x, fp8_dtype=torch.float32)
