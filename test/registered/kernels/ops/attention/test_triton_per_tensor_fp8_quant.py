import importlib.util
import sys
from pathlib import Path

import pytest
import torch

# Optional imports: keep this module collectable in a minimal environment where
# the full sglang package (and its CI helpers) is not installed.
try:
    from sglang.kernels.jit.utils import get_ci_test_range
except Exception:  # noqa: BLE001

    def get_ci_test_range(full, ci):
        return ci


def _load_quant_module():
    """Load the kernel by file path so importing it does not pull in sglang."""
    here = Path(__file__).resolve()
    repo_root = here.parents[5]
    target = (
        repo_root
        / "python/sglang/kernels/ops/attention/triton_per_tensor_fp8_quant.py"
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
    except Exception:  # noqa: BLE001
        sys.modules.pop(name, None)
        return None
    return module


_quant_mod = _load_quant_module()
fused_per_tensor_fp8_quant = (
    None if _quant_mod is None else _quant_mod.fused_per_tensor_fp8_quant
)
_BLOCK = None if _quant_mod is None else getattr(_quant_mod, "_BLOCK", 4096)
_PARTS = None if _quant_mod is None else getattr(_quant_mod, "_PARTS", 256)
_MIN_DESCALE = (
    None if _quant_mod is None else getattr(_quant_mod, "_MIN_DESCALE", 1e-6)
)


try:
    from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

    register_cuda_ci(
        est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large"
    )
    register_amd_ci(est_time=60, suite="nightly-amd-kernel-1-gpu", nightly=True)
except Exception:  # noqa: BLE001 - registration is best-effort only
    pass

DEVICE = "cuda"
FP8_DTYPE = torch.float8_e4m3fn


def _require_kernel():
    if fused_per_tensor_fp8_quant is None:
        pytest.skip("triton_per_tensor_fp8_quant unavailable in this environment")
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm device required")


def _reference_quant(x: torch.Tensor):
    amax = (
        x.abs().float().max()
        if x.numel()
        else x.new_zeros((), dtype=torch.float)
    )
    descale = (
        amax / torch.finfo(FP8_DTYPE).max if amax > 0 else amax.new_tensor(_MIN_DESCALE)
    )
    x_fp8 = (x.float() / descale).clamp(-448, 448).to(FP8_DTYPE)
    return x_fp8, descale.reshape(1)


SHAPES = get_ci_test_range(
    [(128, 64, 128), (64, 32, 64)],
    [(128, 64, 128)],
)
NUM_HEADS = get_ci_test_range([8, 16, 32, 64], [16, 32])
BATCH_SIZES = get_ci_test_range(
    [0, 1, 4, 17, 64, 257, 1024, 4096, 16384],
    [0, 1, 64, 1024, 16384],
)


@pytest.mark.parametrize("num_tokens", BATCH_SIZES)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("shape", SHAPES)
def test_correctness(shape, num_heads, num_tokens):
    _require_kernel()
    qk_nope, qk_rope, v_head = shape
    torch.manual_seed(0)
    q = torch.randn(num_tokens, num_heads, qk_nope + qk_rope, device=DEVICE) * 4
    k = torch.randn(num_tokens, num_heads, qk_nope + qk_rope, device=DEVICE)
    kv = torch.randn(num_tokens, num_heads, qk_nope + v_head, device=DEVICE)
    # V is a strided view of the kv_b_proj output, as in MLA prefill.
    v = kv[..., qk_nope:]
    inputs = [x.to(torch.bfloat16) for x in (q, k, v)]
    outputs, descales = fused_per_tensor_fp8_quant(*inputs)
    assert len(outputs) == len(inputs) == len(descales)
    for x, x_fp8, descale in zip(inputs, outputs, descales):
        ref_fp8, ref_descale = _reference_quant(x)
        assert x_fp8.shape == x.shape and x_fp8.is_contiguous()
        assert x_fp8.dtype == FP8_DTYPE
        torch.testing.assert_close(descale, ref_descale, rtol=1e-6, atol=0)
        # x * (1 / descale) may round one e4m3 step away from x / descale.
        ulp_diff = x_fp8.view(torch.int8).int() - ref_fp8.view(torch.int8).int()
        assert ulp_diff.numel() == 0 or ulp_diff.abs().max() <= 1


@pytest.mark.parametrize("num_inputs", [1, 2, 3])
def test_correctness_variable_input_count(num_inputs):
    _require_kernel()
    torch.manual_seed(1)
    s, h, d = 37, 16, 128
    inputs = [
        torch.randn(s, h, d, device=DEVICE, dtype=torch.bfloat16) for _ in range(num_inputs)
    ]
    outputs, descales = fused_per_tensor_fp8_quant(*inputs)
    assert len(outputs) == num_inputs and len(descales) == num_inputs
    for x, x_fp8, descale in zip(inputs, outputs, descales):
        ref_fp8, ref_descale = _reference_quant(x)
        torch.testing.assert_close(descale, ref_descale, rtol=1e-6, atol=0)
        ulp_diff = x_fp8.view(torch.int8).int() - ref_fp8.view(torch.int8).int()
        assert ulp_diff.numel() == 0 or ulp_diff.abs().max() <= 1


@pytest.mark.parametrize("num_tokens", [0, 1, 37])
def test_empty_and_zero_inputs(num_tokens):
    _require_kernel()
    s, h, d = num_tokens, 16, 128
    zeros = torch.zeros(s, h, d, device=DEVICE, dtype=torch.bfloat16)
    outputs, descales = fused_per_tensor_fp8_quant(zeros, zeros, zeros)
    for x_fp8, descale in zip(outputs, descales):
        assert x_fp8.shape == zeros.shape
        assert descale.numel() == 1
        assert torch.isfinite(descale).all()
        assert float(descale) >= _MIN_DESCALE
        assert torch.isfinite(x_fp8.float()).all()


def test_extreme_values_clamped():
    _require_kernel()
    s, h, d = 8, 16, 128
    x = torch.full((s, h, d), 1e4, device=DEVICE, dtype=torch.bfloat16)
    y = -x
    outputs, descales = fused_per_tensor_fp8_quant(x, y)
    for out, descale in zip(outputs, descales):
        assert torch.isfinite(descale).all()
        assert out.float().abs().max() <= 448.0


_STRIPE = None if _BLOCK is None else _PARTS * _BLOCK


@pytest.mark.parametrize(
    "numel",
    [1, 4096, 4097, 4096 * 256 - 1, 4096 * 256, 4096 * 256 + 1, 3 * 4096 * 256 + 7],
)
def test_covers_every_element(numel):
    """One grid serves inputs of different sizes; no element may be dropped."""
    _require_kernel()
    x = torch.full((numel, 1, 1), 0.5, device=DEVICE, dtype=torch.bfloat16)
    expected = 100.0 / torch.finfo(FP8_DTYPE).max
    for pos in sorted({0, 4095, 4096, numel - 1}):
        if pos >= numel:
            continue
        x[pos] = 100.0
        outputs, descales = fused_per_tensor_fp8_quant(x)
        torch.testing.assert_close(
            descales[0], torch.tensor([expected], device=DEVICE), rtol=1e-6, atol=0
        )
        assert outputs[0][pos].float().item() == pytest.approx(
            torch.finfo(FP8_DTYPE).max, rel=1e-2
        )
        x[pos] = 0.5


def test_out_buffer_reused():
    _require_kernel()
    s, h, d = 16, 16, 128
    x = torch.randn(s, h, d, device=DEVICE, dtype=torch.bfloat16)
    out = torch.empty_like(x, dtype=FP8_DTYPE)
    outputs, _ = fused_per_tensor_fp8_quant(x, out=[out])
    assert outputs[0].data_ptr() == out.data_ptr()


def test_rejects_bad_inputs():
    _require_kernel()
    x = torch.randn(4, 4, 4, device=DEVICE, dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        fused_per_tensor_fp8_quant()
    with pytest.raises(ValueError):
        fused_per_tensor_fp8_quant(x, x, x, x)
    with pytest.raises(ValueError):
        fused_per_tensor_fp8_quant(x, fp8_dtype=torch.float8_e4m3fnuz)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
