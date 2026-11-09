import sys

import pytest

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=60, suite="nightly-amd-kernel-1-gpu", nightly=True)

# Import torch/triton and the kernel lazily so this module is collectable (and
# skips cleanly) on CPU-only or triton-less environments instead of erroring at
# import time. The skip is per-test via the ``_require_gpu`` fixture below.
try:
    import torch
except ImportError:  # pragma: no cover - torch is a hard dep in CI
    torch = None

_HAS_GPU = torch is not None and torch.cuda.is_available()
_HAS_TRITON = False
if _HAS_GPU:
    try:
        import triton  # noqa: F401

        _HAS_TRITON = True
    except ImportError:
        _HAS_TRITON = False

DEVICE = "cuda"


def _require_gpu() -> None:
    if not _HAS_GPU:
        pytest.skip("mla_kv_pack_quantize_fp8 requires a CUDA/ROCm device")
    if not _HAS_TRITON:
        pytest.skip("mla_kv_pack_quantize_fp8 requires triton")


def _kernel():
    from sglang.kernels.ops.attention.mla_kv_pack_quantize_fp8 import (
        mla_kv_pack_quantize_fp8,
    )

    return mla_kv_pack_quantize_fp8


FP8_DTYPE = torch.float8_e4m3fn if torch is not None else None


def _reference(x: torch.Tensor, scale_inv: float) -> torch.Tensor:
    return (x.float() * scale_inv).clamp(-448, 448).to(FP8_DTYPE)


@pytest.mark.parametrize("s", [0, 1, 7, 33, 129])
@pytest.mark.parametrize("num_heads", [1, 12])
def test_mla_kv_pack_quantize_fp8_matches_reference(s: int, num_heads: int) -> None:
    _require_gpu()
    mla_kv_pack_quantize_fp8 = _kernel()
    torch.manual_seed(0)
    qk_nope, qk_rope, v_head = 128, 64, 128
    k_nope = torch.randn(s, num_heads, qk_nope, device=DEVICE, dtype=torch.bfloat16)
    k_pe = torch.randn(s, 1, qk_rope, device=DEVICE, dtype=torch.bfloat16)
    v = torch.randn(s, num_heads, v_head, device=DEVICE, dtype=torch.bfloat16)
    k_scale_inv, v_scale_inv = 0.5, 0.25
    k_out, v_out = mla_kv_pack_quantize_fp8(
        k_nope, k_pe, v, k_scale_inv, v_scale_inv, fp8_dtype=FP8_DTYPE
    )
    assert k_out.shape == (s, num_heads, qk_nope + qk_rope)
    assert v_out.shape == (s, num_heads, v_head)
    if s == 0:
        return
    ref_k = torch.cat(
        [
            _reference(k_nope, k_scale_inv),
            _reference(k_pe, k_scale_inv).expand(-1, num_heads, -1),
        ],
        dim=-1,
    )
    ref_v = _reference(v, v_scale_inv)
    torch.testing.assert_close(k_out.float(), ref_k.float(), rtol=0, atol=0)
    torch.testing.assert_close(v_out.float(), ref_v.float(), rtol=0, atol=0)


def test_mla_kv_pack_quantize_fp8_all_zero_and_extreme() -> None:
    _require_gpu()
    mla_kv_pack_quantize_fp8 = _kernel()
    s, num_heads = 16, 4
    k_nope = torch.zeros(s, num_heads, 128, device=DEVICE, dtype=torch.bfloat16)
    k_pe = torch.zeros(s, 1, 64, device=DEVICE, dtype=torch.bfloat16)
    v = torch.zeros(s, num_heads, 128, device=DEVICE, dtype=torch.bfloat16)
    k_out, v_out = mla_kv_pack_quantize_fp8(k_nope, k_pe, v, 1.0, 1.0, fp8_dtype=FP8_DTYPE)
    assert torch.isfinite(k_out.float()).all()
    assert torch.isfinite(v_out.float()).all()

    k_nope.fill_(1e4)
    v.fill_(-1e4)
    k_out, v_out = mla_kv_pack_quantize_fp8(k_nope, k_pe, v, 1.0, 1.0, fp8_dtype=FP8_DTYPE)
    assert k_out.float().abs().max() <= 448
    assert v_out.float().abs().max() <= 448


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
