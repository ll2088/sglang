import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=30, suite="nightly-amd-kernel-1-gpu", nightly=True)


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
    fn = _import_quant()
    x = torch.zeros((1, 1, 1), dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        fn(x, fp8_dtype=torch.float32)


def test_rejects_non_3d():
    fn = _import_quant()
    x = torch.zeros((4, 8), dtype=torch.bfloat16)
    with pytest.raises(AssertionError):
        fn(x)


def test_rejects_non_contiguous_inner_dim():
    fn = _import_quant()
    x = torch.zeros((4, 8, 16), dtype=torch.bfloat16).transpose(1, 2)
    assert x.stride(2) != 1
    with pytest.raises(AssertionError):
        fn(x)


def test_rejects_mixed_devices():
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


def test_registry_query_is_stable():
    from sglang.kernels.registry import registry

    ops = registry.ops()
    assert isinstance(ops, list)
    assert ops == sorted(ops)
    assert registry.ops() == ops


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
