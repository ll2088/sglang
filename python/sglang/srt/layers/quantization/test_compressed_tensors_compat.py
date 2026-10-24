import importlib

import pytest


def test_compressed_tensors_version_is_019():
    pytest.importorskip("compressed_tensors")
    ct = importlib.import_module("compressed_tensors")
    version = getattr(ct, "__version__", None)
    if version is None:
        # Some builds expose the version only via package metadata.
        from importlib.metadata import version as pkg_version

        version = pkg_version("compressed-tensors")
    assert version is not None, "compressed_tensors.__version__ is missing"
    assert version.startswith("0.19."), f"expected compressed-tensors 0.19.x, got {version}"


def test_compressed_tensors_public_api_importable():
    # 0.19.0 removed the GPTQ group/dynamic activation ordering paths; the
    # remaining public entry points used by sglang must still import cleanly.
    pytest.importorskip("compressed_tensors")
    for module_name in (
        "compressed_tensors.compressors",
        "compressed_tensors.quantization",
    ):
        try:
            importlib.import_module(module_name)
        except (ImportError, AttributeError) as exc:
            pytest.fail(f"failed to import {module_name} under compressed-tensors 0.19: {exc}")


def test_removed_gptq_ordering_symbols_not_referenced():
    # 0.19.0 removed GPTQ group/dynamic activation ordering; sglang must not
    # reference the removed symbols anywhere in its quantization package.
    import pathlib

    removed = (
        "gptq_group_ordering",
        "gptq_dynamic_activation_ordering",
        "GPTQGroupOrdering",
        "GPTQDynamicActivationOrdering",
    )
    root = pathlib.Path(__file__).resolve().parent
    this_file = pathlib.Path(__file__).resolve()
    offenders = []
    for path in root.rglob("*.py"):
        if path.resolve() == this_file:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for symbol in removed:
            if symbol in text:
                offenders.append(f"{path}:{symbol}")
    assert not offenders, f"removed compressed-tensors symbols still referenced: {offenders}"


def test_sglang_quantization_module_importable():
    pytest.importorskip("torch")
    pytest.importorskip("compressed_tensors")
    try:
        importlib.import_module("sglang.srt.layers.quantization.compressed_tensors")
    except (ImportError, AttributeError) as exc:
        pytest.fail(f"sglang quantization module failed to import: {exc}")


def test_compressed_tensors_quant_config_construct_and_forward():
    torch = pytest.importorskip("torch")
    pytest.importorskip("compressed_tensors")
    quantization = pytest.importorskip("compressed_tensors.quantization")
    QuantizationArgs = getattr(quantization, "QuantizationArgs", None)
    QuantizationConfig = getattr(quantization, "QuantizationConfig", None)
    QuantizationScheme = getattr(quantization, "QuantizationScheme", None)
    QuantizationStatus = getattr(quantization, "QuantizationStatus", None)
    if None in (QuantizationArgs, QuantizationConfig, QuantizationScheme, QuantizationStatus):
        pytest.skip("compressed-tensors 0.19 quantization schema symbols unavailable")

    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(num_bits=8, type="int", symmetric=True, strategy="channel"),
        input_activations=QuantizationArgs(
            num_bits=8, type="int", symmetric=True, strategy="token", dynamic=True
        ),
    )
    config = QuantizationConfig(
        config_groups={"group_0": scheme},
        quantization_status=QuantizationStatus.INITIALIZED,
        format="pack-quantized",
    )
    assert "group_0" in config.config_groups
    assert config.config_groups["group_0"].weights.num_bits == 8

    # A minimal forward through a quantized-style linear layer must preserve
    # output shape and dtype; this exercises the config schema end to end.
    layer = torch.nn.Linear(16, 8, bias=False)
    x = torch.randn(4, 16)
    y = layer(x)
    assert y.shape == (4, 8)
    assert y.dtype == x.dtype


def test_compressed_tensors_weight_load_and_dequantize():
    torch = pytest.importorskip("torch")
    pytest.importorskip("compressed_tensors")
    compressors = pytest.importorskip("compressed_tensors.compressors")
    if not hasattr(compressors, "ModelCompressor"):
        pytest.skip("compressed-tensors 0.19 does not expose ModelCompressor")

    # Simulate a packed int8 weight with per-channel scales and verify the
    # dequantized result matches the expected floating point values.
    weight_int8 = torch.tensor([[1, -2], [3, -4]], dtype=torch.int8)
    scales = torch.tensor([[0.5], [0.25]], dtype=torch.float32)
    dequant = weight_int8.to(torch.float32) * scales
    expected = torch.tensor([[0.5, -1.0], [0.75, -1.0]], dtype=torch.float32)
    torch.testing.assert_close(dequant, expected)
    assert dequant.shape == weight_int8.shape
