import importlib

import pytest


def test_compressed_tensors_version_is_019():
    ct = importlib.import_module("compressed_tensors")
    version = getattr(ct, "__version__", None)
    assert version is not None, "compressed_tensors.__version__ is missing"
    assert version.startswith("0.19."), f"expected compressed-tensors 0.19.x, got {version}"


def test_compressed_tensors_public_api_importable():
    # 0.19.0 removed the GPTQ group/dynamic activation ordering paths; the
    # remaining public entry points used by sglang must still import cleanly.
    for module_name in (
        "compressed_tensors.compressors",
        "compressed_tensors.quantization",
    ):
        importlib.import_module(module_name)


def test_sglang_quantization_module_importable():
    pytest.importorskip("torch")
    importlib.import_module("sglang.srt.layers.quantization.compressed_tensors")
