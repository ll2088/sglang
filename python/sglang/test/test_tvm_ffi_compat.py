"""Compatibility checks for the apache-tvm-ffi 0.1.15.post2 upgrade.

These tests are intentionally lightweight: they only import
``sglang._platform_stubs`` (stdlib-only) and ``sglang.kernels.jit`` (an empty
package marker), so they run on a CPU-only box without torch or a GPU.
"""

import importlib
import importlib.metadata

import pytest

from sglang._platform_stubs import (
    _parse_tvm_ffi_version,
    require_tvm_ffi_version,
)

MINIMUM_TVM_FFI = "0.1.15.post2"


def test_parse_tvm_ffi_post_version():
    """``0.1.15.post2`` must parse as release (0, 1, 15) with post 2."""
    assert _parse_tvm_ffi_version("0.1.15.post2") == ((0, 1, 15), 2)
    assert _parse_tvm_ffi_version("0.1.15") == ((0, 1, 15), 0)
    assert _parse_tvm_ffi_version("0.1.11") < _parse_tvm_ffi_version(
        MINIMUM_TVM_FFI
    )
    # The post suffix must order above the plain release it follows.
    assert _parse_tvm_ffi_version("0.1.15") < _parse_tvm_ffi_version(
        "0.1.15.post2"
    )
    assert _parse_tvm_ffi_version("0.1.15.post1") < _parse_tvm_ffi_version(
        "0.1.15.post2"
    )


def test_require_tvm_ffi_version_missing(monkeypatch):
    """A missing install must raise a clear error, not silently degrade."""
    monkeypatch.setattr(
        "sglang._platform_stubs._tvm_ffi_version", lambda: None
    )
    with pytest.raises(RuntimeError, match="not installed"):
        require_tvm_ffi_version(MINIMUM_TVM_FFI)
    # The error must name the minimum so users know what to install.
    with pytest.raises(RuntimeError, match=MINIMUM_TVM_FFI):
        require_tvm_ffi_version(MINIMUM_TVM_FFI)


def test_require_tvm_ffi_version_too_old(monkeypatch):
    """An older install must raise a clear error naming the minimum."""
    monkeypatch.setattr(
        "sglang._platform_stubs._tvm_ffi_version", lambda: "0.1.11"
    )
    with pytest.raises(RuntimeError, match="older than the required"):
        require_tvm_ffi_version(MINIMUM_TVM_FFI)
    # A post release below the minimum must also be rejected.
    monkeypatch.setattr(
        "sglang._platform_stubs._tvm_ffi_version", lambda: "0.1.15.post1"
    )
    with pytest.raises(RuntimeError, match="older than the required"):
        require_tvm_ffi_version(MINIMUM_TVM_FFI)


def test_require_tvm_ffi_version_ok(monkeypatch):
    monkeypatch.setattr(
        "sglang._platform_stubs._tvm_ffi_version", lambda: MINIMUM_TVM_FFI
    )
    assert require_tvm_ffi_version(MINIMUM_TVM_FFI) == MINIMUM_TVM_FFI


def test_kernels_jit_importable():
    """``sglang.kernels.jit`` must import without pulling in a kernel backend."""
    module = importlib.import_module("sglang.kernels.jit")
    assert module is not None
    # The package marker must be a real module object, not a lazy proxy.
    assert getattr(module, "__name__", None) == "sglang.kernels.jit"


def test_installed_tvm_ffi_meets_minimum():
    """The installed apache-tvm-ffi must satisfy the pinned minimum."""
    try:
        version = importlib.metadata.version("apache-tvm-ffi")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("apache-tvm-ffi is not installed in this environment")
    assert _parse_tvm_ffi_version(version) >= _parse_tvm_ffi_version(
        MINIMUM_TVM_FFI
    )


def test_check_env_reports_tvm_ffi_version():
    """``check_env`` must report the installed apache-tvm-ffi version."""
    from sglang.check_env import BaseEnv

    versions = BaseEnv().get_package_versions()
    assert "apache-tvm-ffi" in versions
    reported = versions["apache-tvm-ffi"]
    if reported == "Module Not Found":
        pytest.skip("apache-tvm-ffi is not installed in this environment")
    # The reported string must parse and satisfy the pinned minimum, so a
    # post-suffix version like 0.1.15.post2 is not misjudged.
    assert _parse_tvm_ffi_version(reported) >= _parse_tvm_ffi_version(
        MINIMUM_TVM_FFI
    )
