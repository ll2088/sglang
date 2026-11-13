"""Compatibility checks for the apache-tvm-ffi dependency bump (0.1.11 -> 0.1.16).

apache-tvm-ffi is only declared in python/pyproject.toml; SGLang does not call
its Python API directly. These tests pin the expected version and make sure the
key SGLang modules still import cleanly after the upgrade.
"""

import importlib
import importlib.metadata

import pytest

EXPECTED_TVM_FFI_VERSION = "0.1.16"


def test_tvm_ffi_version_is_pinned():
    """The installed apache-tvm-ffi must match the pinned version."""
    try:
        installed = importlib.metadata.version("apache-tvm-ffi")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("apache-tvm-ffi is not installed in this environment")
    assert installed == EXPECTED_TVM_FFI_VERSION, (
        f"expected apache-tvm-ffi=={EXPECTED_TVM_FFI_VERSION}, got {installed}"
    )


def test_tvm_ffi_importable():
    """The tvm_ffi package itself must import without errors."""
    try:
        importlib.import_module("tvm_ffi")
    except ImportError:
        pytest.skip("tvm_ffi is not installed in this environment")


@pytest.mark.parametrize(
    "module_name",
    [
        "sglang",
        "sglang.kernels",
        "sglang.kernels.jit",
        "sglang._platform_stubs",
    ],
)
def test_sglang_modules_importable(module_name):
    """Key SGLang modules must remain importable after the dependency bump."""
    importlib.import_module(module_name)
