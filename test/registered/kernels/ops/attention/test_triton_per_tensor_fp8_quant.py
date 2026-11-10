"""Lightweight tests for the fused per-tensor FP8 quantizer.

Loaded by file path so they run without a full sglang install. torch and triton
are optional: the module skips cleanly when either is missing, so CI jobs that
only install pytest do not error during collection.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="torch required for FP8 quant tests")
pytest.importorskip("triton", reason="triton required for FP8 quant tests")


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

pytestmark = pytest.mark.skipif(
    _quant is None, reason="triton_per_tensor_fp8_quant unavailable"
)


def _has_gpu():
    return torch.cuda.is_available()


requires_gpu = pytest.mark.skipif(
    not _has_gpu(), reason="CUDA/ROCm device required"
)


def _reference_quant(x, fp8_dtype):
    fp8_max = float(torch.finfo(fp8_dtype).max)
    amax = x.abs().float().max() if x.numel() else x.new_zeros((), dtype=torch.float)
    descale = amax / fp8_max if amax > 0 else amax.new_tensor(1e-6)
    x_fp8 = (x.float() / descale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    return x_fp8, descale.reshape(1)


@requires_gpu
@pytest.mark.parametrize("num_tokens", [0, 1, 37, 8192])
def test_fused_per_tensor_fp8_quant_matches_reference(num_tokens):
    torch.manual_seed(0)
    q = torch.randn(num_tokens, 12, 192, device="cuda") * 4
    k = torch.randn(num_tokens, 12, 192, device="cuda")
    kv = torch.randn(num_tokens, 12, 256, device="cuda")
    v = kv[..., 128:]
    inputs = [x.to(torch.bfloat16) for x in (q, k, v)]
    outputs, descales = _quant.fused_per_tensor_fp8_quant(*inputs)
    for x, x_fp8, descale in zip(inputs, outputs, descales):
        ref_fp8, ref_descale = _reference_quant(x, torch.float8_e4m3fn)
        assert x_fp8.shape == x.shape and x_fp8.is_contiguous()
        torch.testing.assert_close(descale, ref_descale, rtol=1e-6, atol=0)
        ulp_diff = x_fp8.view(torch.int8).int() - ref_fp8.view(torch.int8).int()
        assert ulp_diff.numel() == 0 or ulp_diff.abs().max() <= 1


@requires_gpu
def test_fused_per_tensor_fp8_quant_empty_and_zero_inputs():
    empty = torch.empty(0, 4, 128, dtype=torch.bfloat16, device="cuda")
    zeros = torch.zeros(16, 4, 128, dtype=torch.bfloat16, device="cuda")
    outputs, descales = _quant.fused_per_tensor_fp8_quant(empty, zeros)
    assert outputs[0].numel() == 0
    assert outputs[1].shape == zeros.shape
    for d in descales:
        assert d.numel() == 1 and float(d) > 0
    assert torch.count_nonzero(outputs[1].float()) == 0


@requires_gpu
def test_fused_per_tensor_fp8_quant_rejects_bad_inputs():
    x = torch.randn(4, 2, 64, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError):
        _quant.fused_per_tensor_fp8_quant()
    with pytest.raises(ValueError):
        _quant.fused_per_tensor_fp8_quant(x, x, x, x)
    with pytest.raises(ValueError):
        _quant.fused_per_tensor_fp8_quant(x, fp8_dtype=torch.float32)


@requires_gpu
def test_fused_per_tensor_fp8_quant_accepts_out_buffers():
    x = torch.randn(8, 2, 64, dtype=torch.bfloat16, device="cuda")
    out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    outputs, _ = _quant.fused_per_tensor_fp8_quant(x, out=[out])
    assert outputs[0].data_ptr() == out.data_ptr()
