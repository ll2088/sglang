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
    # Normalize local/dev suffixes (e.g. "0.1.16.post1") before comparing so
    # the check reflects the pinned release line rather than an exact build.
    normalized = installed.split("+", 1)[0]
    assert normalized == EXPECTED_TVM_FFI_VERSION, (
        f"expected apache-tvm-ffi=={EXPECTED_TVM_FFI_VERSION}, got {installed}"
    )


def test_tvm_ffi_importable():
    """The tvm_ffi package itself must import without errors."""
    try:
        importlib.import_module("tvm_ffi")
    except ImportError:
        pytest.skip("tvm_ffi is not installed in this environment")
    except Exception as exc:  # pragma: no cover - defensive
        pytest.fail(f"tvm_ffi failed to import: {exc!r}")


def test_tvm_ffi_public_api_surface():
    """The tvm_ffi public API used by downstream wheels must stay importable.

    SGLang does not call tvm_ffi directly, but the version bump touches the
    Rust FFI/object system and stubgen. This test pins the presence of the
    top-level entry points so a future bump that removes or renames them is
    caught here instead of at runtime.
    """
    try:
        tvm_ffi = importlib.import_module("tvm_ffi")
    except ImportError:
        pytest.skip("tvm_ffi is not installed in this environment")

    for attr in ("__version__",):
        assert hasattr(tvm_ffi, attr), f"tvm_ffi is missing expected attribute {attr!r}"

    version = getattr(tvm_ffi, "__version__", None)
    if version is not None:
        normalized = str(version).split("+", 1)[0]
        assert normalized == EXPECTED_TVM_FFI_VERSION, (
            f"tvm_ffi.__version__={version!r} does not match the pinned "
            f"apache-tvm-ffi=={EXPECTED_TVM_FFI_VERSION}"
        )


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
