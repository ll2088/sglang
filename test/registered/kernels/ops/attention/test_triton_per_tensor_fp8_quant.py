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

NUM_HEADS = get_ci_test_range([1, 8, 16, 32], [8, 16])
TOKEN_COUNTS = get_ci_test_range(
    [0, 1, 37, 1024, 8192],
    [0, 1, 37, 8192],
)


def _skip_if_unavailable():
    if fused_per_tensor_fp8_quant is None:
        pytest.skip("triton_per_tensor_fp8_quant unavailable in this environment")
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm device required")


def _reference_quant(x: torch.Tensor, fp8_dtype: torch.dtype = FP8_DTYPE):
    fp8_max = float(torch.finfo(fp8_dtype).max)
    if x.numel():
        amax = x.abs().float().max()
    else:
        amax = x.new_zeros((), dtype=torch.float)
    descale = amax / fp8_max if amax > 0 else amax.new_tensor(_MIN_DESCALE)
    x_fp8 = (x.float() / descale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    return x_fp8, descale.reshape(1)


@pytest.mark.parametrize("num_tokens", TOKEN_COUNTS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
def test_correctness(num_tokens, num_heads):
    _skip_if_unavailable()
    torch.manual_seed(0)
    qk_dim, v_dim = 192, 128
    q = torch.randn(num_tokens, num_heads, qk_dim, device=DEVICE) * 4
    k = torch.randn(num_tokens, num_heads, qk_dim, device=DEVICE)
    kv = torch.randn(num_tokens, num_heads, 128 + v_dim, device=DEVICE)
    # V is a strided view of the kv_b_proj output, as in MLA prefill.
    v = kv[..., 128:]
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


def test_single_and_two_inputs():
    _skip_if_unavailable()
    torch.manual_seed(0)
    x = torch.randn(64, 8, 128, device=DEVICE, dtype=torch.bfloat16)
    y = torch.randn(64, 8, 128, device=DEVICE, dtype=torch.bfloat16)

    outs1, scales1 = fused_per_tensor_fp8_quant(x)
    assert len(outs1) == 1 and len(scales1) == 1
    ref_fp8, ref_descale = _reference_quant(x)
    torch.testing.assert_close(scales1[0], ref_descale, rtol=1e-6, atol=0)
    ulp_diff = outs1[0].view(torch.int8).int() - ref_fp8.view(torch.int8).int()
    assert ulp_diff.abs().max() <= 1

    outs2, scales2 = fused_per_tensor_fp8_quant(x, y)
    assert len(outs2) == 2 and len(scales2) == 2
    for src, out, scale in zip((x, y), outs2, scales2):
        ref_fp8, ref_descale = _reference_quant(src)
        torch.testing.assert_close(scale, ref_descale, rtol=1e-6, atol=0)
        ulp_diff = out.view(torch.int8).int() - ref_fp8.view(torch.int8).int()
        assert ulp_diff.abs().max() <= 1


def test_empty_and_all_zero_inputs():
    _skip_if_unavailable()
    empty = torch.empty(0, 8, 128, device=DEVICE, dtype=torch.bfloat16)
    zeros = torch.zeros(16, 8, 128, device=DEVICE, dtype=torch.bfloat16)
    normal = torch.randn(16, 8, 128, device=DEVICE, dtype=torch.bfloat16)

    outs, scales = fused_per_tensor_fp8_quant(empty, zeros, normal)
    assert outs[0].numel() == 0
    # Empty and all-zero inputs get a positive descale so downstream kernels
    # never divide by zero.
    assert float(scales[0]) > 0 and float(scales[1]) > 0
    assert torch.all(outs[1] == 0)
    ref_fp8, ref_descale = _reference_quant(normal)
    torch.testing.assert_close(scales[2], ref_descale, rtol=1e-6, atol=0)
    ulp_diff = outs[2].view(torch.int8).int() - ref_fp8.view(torch.int8).int()
    assert ulp_diff.abs().max() <= 1


def test_extreme_values_are_clamped():
    _skip_if_unavailable()
    x = torch.full((8, 4, 128), 1e4, device=DEVICE, dtype=torch.bfloat16)
    x[0, 0, 0] = -1e4
    outs, scales = fused_per_tensor_fp8_quant(x)
    fp8_max = float(torch.finfo(FP8_DTYPE).max)
    assert float(scales[0]) > 0
    assert outs[0].float().abs().max() <= fp8_max
    # The largest magnitude element must map to the FP8 max, not inf/NaN.
    assert torch.isfinite(outs[0].float()).all()


def test_strided_inputs():
    _skip_if_unavailable()
    torch.manual_seed(0)
    s, h = 16, 8
    full = torch.randn(s, h, 256, device=DEVICE, dtype=torch.bfloat16)
    view = full[..., 128:]
    assert view.stride(-1) == 1
    outs, scales = fused_per_tensor_fp8_quant(view)
    ref_fp8, ref_descale = _reference_quant(view)
    torch.testing.assert_close(scales[0], ref_descale, rtol=1e-6, atol=0)
    ulp_diff = outs[0].view(torch.int8).int() - ref_fp8.view(torch.int8).int()
    assert ulp_diff.abs().max() <= 1


def test_out_buffer_reuse():
    _skip_if_unavailable()
    torch.manual_seed(0)
    x = torch.randn(32, 8, 128, device=DEVICE, dtype=torch.bfloat16)
    out = torch.empty_like(x, dtype=FP8_DTYPE)
    outs, _ = fused_per_tensor_fp8_quant(x, out=[out])
    assert outs[0].data_ptr() == out.data_ptr()


def test_rejects_bad_inputs():
    _skip_if_unavailable()
    x = torch.randn(4, 8, 128, device=DEVICE, dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        fused_per_tensor_fp8_quant()
    with pytest.raises(ValueError):
        fused_per_tensor_fp8_quant(x, x, x, x)
    with pytest.raises(ValueError):
        fused_per_tensor_fp8_quant(x, fp8_dtype=torch.float32)


def test_covers_every_element():
    """One grid serves inputs of different sizes; no element may be dropped."""
    _skip_if_unavailable()
    stripe = _PARTS * _BLOCK
    for numel in (1, _BLOCK, _BLOCK + 1, stripe - 1, stripe, stripe + 1):
        x = torch.full((numel, 1, 1), 0.5, device=DEVICE, dtype=torch.bfloat16)
        outs, scales = fused_per_tensor_fp8_quant(x)
        expected = 0.5 / float(scales[0])
        got = outs[0].float().flatten()
        assert got.numel() == numel
        assert torch.allclose(
            got, torch.full_like(got, expected), rtol=1e-2, atol=1e-2
        ), f"numel={numel}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
