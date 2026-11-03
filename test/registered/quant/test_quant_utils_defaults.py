# SPDX-License-Identifier: Apache-2.0
"""Regression tests for mutable-default and module-level mapping isolation.

Mirrors the upstream fix pattern: mutable defaults must be ``None`` and
module-level dicts must never be mutated in place by callers.
"""

import importlib.util
import sys
import types
from pathlib import Path
from types import MappingProxyType

import pytest


def _load_quant_utils():
    """Load the quantization utils source file without importing sglang.

    Importing ``sglang.srt.layers.quantization.utils`` pulls in the full
    runtime (CUDA kernels, sgl_kernel, ...).  The module only needs a few
    third-party symbols at import time, so we stub them and load the file
    directly via importlib.
    """
    repo_root = Path(__file__).resolve().parents[3]
    source = repo_root / "python/sglang/srt/layers/quantization/utils.py"
    if not source.is_file():
        pytest.skip(f"quantization utils source not found at {source}")

    # Stub the fp8 kernel import used at module import time.
    fp8_kernel = types.ModuleType("sglang.kernels.ops.quantization.fp8_kernel")
    fp8_kernel.scaled_fp8_quant = lambda *args, **kwargs: (None, None)
    sys.modules.setdefault(
        "sglang.kernels.ops.quantization.fp8_kernel", fp8_kernel
    )

    spec = importlib.util.spec_from_file_location(
        "_sglang_quant_utils_under_test", source
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


quant_utils = _load_quant_utils()


def test_fallback_fused_shards_is_not_mutated_by_callers():
    original = {key: list(value) for key, value in quant_utils._FALLBACK_FUSED_SHARDS.items()}
    copied = quant_utils._copy_fused_mapping(None)
    copied["qkv_proj"].append("extra_proj")
    copied["new_key"] = ["x"]
    assert quant_utils._FALLBACK_FUSED_SHARDS == original


def test_copy_fused_mapping_returns_fresh_lists():
    first = quant_utils._copy_fused_mapping(None)
    second = quant_utils._copy_fused_mapping(None)
    assert first is not second
    assert first["qkv_proj"] is not second["qkv_proj"]
    first["qkv_proj"].append("mutated")
    assert "mutated" not in second["qkv_proj"]


def test_copy_fused_mapping_does_not_mutate_caller_dict():
    caller = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
    snapshot = {key: list(value) for key, value in caller.items()}
    copied = quant_utils._copy_fused_mapping(caller)
    copied["qkv_proj"].append("extra")
    assert caller == snapshot


def test_copy_fused_mapping_rejects_non_mapping():
    with pytest.raises(TypeError):
        quant_utils._copy_fused_mapping(["qkv_proj"])


def test_is_layer_skipped_does_not_mutate_fallback():
    original = {key: list(value) for key, value in quant_utils._FALLBACK_FUSED_SHARDS.items()}
    quant_utils.is_layer_skipped(
        "model.layers.0.self_attn.qkv_proj",
        ignored_layers=[],
        fused_mapping=None,
    )
    assert quant_utils._FALLBACK_FUSED_SHARDS == original


def test_mapping_proxy_source_is_read_only():
    proxy = MappingProxyType(quant_utils._FALLBACK_FUSED_SHARDS)
    with pytest.raises(TypeError):
        proxy["qkv_proj"] = []
