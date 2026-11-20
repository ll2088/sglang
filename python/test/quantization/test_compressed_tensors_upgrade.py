# SPDX-License-Identifier: Apache-2.0
"""Compatibility tests for the compressed-tensors 0.18.0 -> 0.20.1 upgrade.

These tests only exercise the sgl_kernel quantization surface that the
upgrade touches. They are written to be import-safe on CPU-only machines:
anything that needs a compiled CUDA extension is skipped instead of failing.
"""

import importlib
import importlib.util

import pytest


def _import_or_skip(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"{module_name} is not importable: {exc}")
    except (RuntimeError, OSError) as exc:  # pragma: no cover - env dependent
        # Compiled CUDA extensions may fail to load (missing libcudart, wrong
        # arch, etc.). Treat that as an environment limitation, not a
        # regression of the compressed-tensors upgrade.
        pytest.skip(f"{module_name} failed to load: {exc}")


def test_quantization_module_imports_and_exposes_symbols():
    """sgl_kernel.quantization must import and re-export the gguf helpers."""
    quantization = _import_or_skip("sgl_kernel.quantization")
    expected = [
        "ggml_dequantize",
        "ggml_moe_a8",
        "ggml_moe_a8_vec",
        "ggml_moe_get_block_size",
        "ggml_mul_mat_a8",
        "ggml_mul_mat_vec_a8",
    ]
    for name in expected:
        assert hasattr(quantization, name), f"missing symbol: {name}"


def test_scalar_type_mapping_is_stable():
    """scalar_type ids and the float4/float6 entries must stay consistent."""
    scalar_type = _import_or_skip("sgl_kernel.scalar_type")
    ScalarType = scalar_type.ScalarType
    scalar_types = scalar_type.scalar_types

    # Round-trip through the id map used by the C++ side.
    for name in ("uint4b8", "float8_e4m3fn", "float4_e2m1f", "float6_e3m2f"):
        st = getattr(scalar_types, name)
        assert ScalarType.from_id(st.id) == st, name

    # MXFP4 / MXFP6 style formats must remain non-IEEE-754 finite-only types.
    assert scalar_types.float4_e2m1f.is_floating_point()
    assert not scalar_types.float4_e2m1f.has_infs()
    assert scalar_types.float6_e3m2f.is_floating_point()
    assert not scalar_types.float6_e3m2f.has_infs()


def test_load_utils_exposes_loader_entrypoints():
    """load_utils must keep the loader helpers used by the quantization path."""
    load_utils = _import_or_skip("sgl_kernel.load_utils")
    for name in (
        "_get_compute_capability",
        "_filter_compiled_extensions",
        "_load_architecture_specific_ops",
        "_find_cuda_home",
        "_preload_cuda_library",
    ):
        assert callable(getattr(load_utils, name)), f"missing callable: {name}"

    # The compiled-extension filter must prefer .so over plain python files.
    filtered = load_utils._filter_compiled_extensions(
        ["/tmp/common_ops.py", "/tmp/common_ops.abi3.so"]
    )
    assert filtered[0].endswith(".so")


def test_moe_and_gemm_quantization_entrypoints_exist():
    """moe/gemm quantization entrypoints must remain importable and callable."""
    moe = _import_or_skip("sgl_kernel.moe")
    gemm = _import_or_skip("sgl_kernel.gemm")

    for name in ("moe_align_block_size", "topk_softmax", "topk_sigmoid", "moe_sum"):
        assert callable(getattr(moe, name)), f"missing moe entrypoint: {name}"

    for name in (
        "int8_scaled_mm",
        "fp8_scaled_mm",
        "sgl_per_token_group_quant_8bit",
        "sgl_per_token_quant_fp8",
    ):
        assert callable(getattr(gemm, name)), f"missing gemm entrypoint: {name}"

    # The legacy aliases must keep pointing at the 8bit group quant kernel.
    assert gemm.sgl_per_token_group_quant_fp8 is gemm.sgl_per_token_group_quant_8bit
    assert gemm.sgl_per_token_group_quant_int8 is gemm.sgl_per_token_group_quant_8bit


def test_gguf_quantization_helpers_are_wired_to_torch_ops():
    """gguf helpers must forward to the registered sgl_kernel torch ops."""
    torch = pytest.importorskip("torch")
    gguf = _import_or_skip("sgl_kernel.quantization.gguf")

    for name in (
        "ggml_dequantize",
        "ggml_mul_mat_vec_a8",
        "ggml_mul_mat_a8",
        "ggml_moe_a8",
        "ggml_moe_a8_vec",
        "ggml_moe_get_block_size",
    ):
        assert callable(getattr(gguf, name)), f"missing gguf helper: {name}"

    # ggml_dequantize validates its shape arguments before touching the op.
    with pytest.raises(AssertionError):
        gguf.ggml_dequantize(
            torch.empty(0, dtype=torch.uint8), 0, 0, 0, torch.float16
        )
