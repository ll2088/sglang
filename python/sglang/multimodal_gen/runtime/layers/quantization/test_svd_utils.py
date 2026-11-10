# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from sglang.multimodal_gen.runtime.layers.quantization.svd_utils import (
    svd,
    svd_compute_dtype,
)


def _reconstruct(us: torch.Tensor, vt: torch.Tensor) -> torch.Tensor:
    return us @ vt


@pytest.mark.parametrize("dtype,tol", [(torch.float32, 1e-4), (torch.float64, 1e-10)])
def test_svd_dtype_and_reconstruction(dtype, tol):
    torch.manual_seed(0)
    weight = torch.randn(64, 32, dtype=dtype)
    rank = 8
    us, vt = svd(weight, rank)
    assert us.dtype == dtype and vt.dtype == dtype
    assert us.shape == (64, rank) and vt.shape == (rank, 32)
    assert svd_compute_dtype(dtype) == dtype
    err = (_reconstruct(us, vt) - weight).abs().max().item()
    assert err < tol


def test_svd_float32_output_uses_float32_compute():
    weight = torch.randn(16, 8, dtype=torch.bfloat16)
    us, vt = svd(weight, 4)
    assert us.dtype == torch.bfloat16 and vt.dtype == torch.bfloat16
    assert svd_compute_dtype(torch.bfloat16) == torch.float32
    assert svd_compute_dtype(torch.float16) == torch.float32
    assert svd_compute_dtype(torch.float64) == torch.float64
