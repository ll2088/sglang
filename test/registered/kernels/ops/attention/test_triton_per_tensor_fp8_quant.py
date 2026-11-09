import sys

import pytest

# torch is imported lazily inside the tests that need it so that this module
# can be collected in a minimal environment (no GPU, no full sglang deps).
try:
    from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

    register_cuda_ci(
        est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large"
    )
    register_amd_ci(est_time=30, suite="nightly-amd-kernel-1-gpu", nightly=True)
except Exception:  # noqa: BLE001 - CI registration is optional for local runs
    pass


def _torch():
    import torch

    return torch


def _has_cuda() -> bool:
    try:
        return _torch().cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


requires_cuda = pytest.mark.skipif(
    not _has_cuda(), reason="requires a CUDA/ROCm device"
)


def _import_quant():
    from sglang.kernels.ops.attention.triton_per_tensor_fp8_quant import (
        fused_per_tensor_fp8_quant,
    )

    return fused_per_tensor_fp8_quant


def test_import_and_public_api():
    """The module must import without a GPU and expose the public entry point."""
    fn = _import_quant()
    assert callable(fn)


def test_rejects_bad_arity():
    fn = _import_quant()
    with pytest.raises(ValueError):
        fn()


def test_rejects_bad_fp8_dtype():
    torch = _torch()
    fn = _import_quant()
    x = torch.zeros((1, 1, 1), dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        fn(x, fp8_dtype=torch.float32)


def test_rejects_non_3d():
    torch = _torch()
    fn = _import_quant()
    x = torch.zeros((4, 8), dtype=torch.bfloat16)
    with pytest.raises(AssertionError):
        fn(x)


def test_rejects_non_contiguous_inner_dim():
    torch = _torch()
    fn = _import_quant()
    x = torch.zeros((4, 8, 16), dtype=torch.bfloat16).transpose(1, 2)
    assert x.stride(2) != 1
    with pytest.raises(AssertionError):
        fn(x)


def test_rejects_mixed_devices():
    torch = _torch()
    fn = _import_quant()
    a = torch.zeros((1, 1, 1), dtype=torch.bfloat16)
    b = torch.zeros((1, 1, 1), dtype=torch.bfloat16, device="meta")
    with pytest.raises(AssertionError):
        fn(a, b)


def test_min_descale_is_positive():
    """All-zero/empty inputs must get a positive descale to avoid NaNs."""
    from sglang.kernels.ops.attention.triton_per_tensor_fp8_quant import (
        _MIN_DESCALE,
    )

    assert _MIN_DESCALE > 0


def test_rejects_four_inputs():
    torch = _torch()
    fn = _import_quant()
    x = torch.zeros((1, 1, 1), dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        fn(x, x, x, x)


def test_rejects_non_bf16_fp16_input():
    torch = _torch()
    fn = _import_quant()
    x = torch.zeros((1, 1, 1), dtype=torch.float32)
    with pytest.raises(AssertionError):
        fn(x)


def test_rejects_bad_out_shape():
    torch = _torch()
    fn = _import_quant()
    x = torch.zeros((4, 8, 16), dtype=torch.bfloat16)
    bad = torch.empty((4, 8, 15), dtype=torch.float8_e4m3fn)
    with pytest.raises(AssertionError):
        fn(x, out=[bad])


def test_rejects_non_contiguous_out():
    torch = _torch()
    fn = _import_quant()
    x = torch.zeros((4, 8, 16), dtype=torch.bfloat16)
    bad = torch.empty((4, 16, 8), dtype=torch.float8_e4m3fn).transpose(1, 2)
    assert not bad.is_contiguous()
    with pytest.raises(AssertionError):
        fn(x, out=[bad])


def test_accepts_strided_view_input():
    """A strided view with a contiguous inner dim must pass validation."""
    torch = _torch()
    _import_quant()
    full = torch.zeros((4, 8, 32), dtype=torch.bfloat16)
    view = full[..., 16:]
    assert view.stride(2) == 1
    # Validation only; the kernel launch itself needs a GPU.
    from sglang.kernels.ops.attention.triton_per_tensor_fp8_quant import (
        _shape_stride,
    )

    shape, stride = _shape_stride(view)
    assert shape == (8, 16)
    assert stride[2] == 1


def test_registry_query_is_stable():
    try:
        from sglang.kernels.registry import registry
    except Exception as exc:  # noqa: BLE001 - optional in minimal envs
        pytest.skip(f"sglang.kernels.registry unavailable: {exc}")

    ops = registry.ops()
    assert isinstance(ops, list)
    assert ops == sorted(ops)
    assert registry.ops() == ops


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
