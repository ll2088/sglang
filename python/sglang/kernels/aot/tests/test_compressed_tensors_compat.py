# SPDX-License-Identifier: Apache-2.0
"""Compatibility checks for the compressed-tensors 0.20.1 upgrade.

These tests only exercise the sgl_kernel quantization surface that the
repository owns; they do not import compressed-tensors itself so they stay
runnable in environments where the optional dependency is absent.
"""

import importlib

import pytest

pytest.importorskip("torch")


def test_quantization_module_imports_and_exports():
    module = importlib.import_module("sgl_kernel.quantization")
    for symbol in (
        "ggml_dequantize",
        "ggml_mul_mat_vec_a8",
        "ggml_mul_mat_a8",
        "ggml_moe_a8",
        "ggml_moe_a8_vec",
        "ggml_moe_get_block_size",
    ):
        assert hasattr(module, symbol), f"missing symbol: {symbol}"


def test_scalar_type_gguf_mapping_is_stable():
    pytest.importorskip("torch")
    from sgl_kernel.scalar_type import NanRepr, ScalarType, scalar_types

    # GGUF / GPTQ style integer types used by the quantization loaders.
    assert scalar_types.uint4b8 == ScalarType.uint(4, 8)
    assert scalar_types.uint8b128 == ScalarType.uint(8, 128)
    assert scalar_types.uint2b2 == ScalarType.uint(2, 2)
    assert scalar_types.uint3b4 == ScalarType.uint(3, 4)

    # MXFP-style floating point types referenced by the new presets.
    assert scalar_types.float4_e2m1f == ScalarType.float_(2, 1, True, NanRepr.NONE)
    assert scalar_types.float6_e3m2f == ScalarType.float_(3, 2, True, NanRepr.NONE)

    # Round-trip through the id map used by the custom ops.
    for name in ("uint4b8", "uint8b128", "float4_e2m1f", "float6_e3m2f"):
        st = getattr(scalar_types, name)
        assert ScalarType.from_id(st.id) == st


def test_load_utils_importable():
    pytest.importorskip("torch")
    module = importlib.import_module("sgl_kernel.load_utils")
    assert callable(module._get_compute_capability)
    assert callable(module._filter_compiled_extensions)
    assert callable(module._load_architecture_specific_ops)


def test_moe_and_gemm_modules_importable():
    pytest.importorskip("torch")
    moe = importlib.import_module("sgl_kernel.moe")
    gemm = importlib.import_module("sgl_kernel.gemm")
    assert callable(moe.moe_align_block_size)
    assert callable(moe.topk_softmax)
    assert callable(gemm.int8_scaled_mm)
    assert callable(gemm.fp8_scaled_mm)
    assert callable(gemm.sgl_per_token_group_quant_8bit)


def test_gemm_quant_dispatch_requires_cuda():
    torch = pytest.importorskip("torch")

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the quantized GEMM kernels")

    from sgl_kernel.gemm import sgl_per_token_group_quant_8bit

    x = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
    out_q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    out_s = torch.empty(4, 1, device="cuda", dtype=torch.float32)
    sgl_per_token_group_quant_8bit(x, out_q, out_s, 128, 1e-4, -448.0, 448.0)
    assert out_q.shape == x.shape
    assert out_s.shape == (4, 1)
