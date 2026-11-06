"""Compatibility tests for the apache-tvm-ffi dependency bump.

These tests pin the Python-side contract sglang relies on after upgrading
apache-tvm-ffi from 0.1.11 to 0.1.15.post2:

* the ``sglang.kernels.jit`` import path still works and tvm-ffi is importable;
* the installed version is at least the pinned minimum;
* ``_platform_stubs`` parses ``post``-suffixed versions correctly and fails
  loudly (instead of silently degrading) when tvm-ffi is missing or too old.

They are written to be runnable without a GPU and without importing heavy
sglang runtime modules.
"""

import importlib
import importlib.metadata
import sys

import pytest

from sglang import _platform_stubs

MINIMUM_TVM_FFI = "0.1.15.post2"


def _installed_tvm_ffi_version():
    try:
        return importlib.metadata.version("apache-tvm-ffi")
    except importlib.metadata.PackageNotFoundError:
        return None


def test_kernels_jit_imports_and_tvm_ffi_available():
    """Importing sglang.kernels.jit must not break, and tvm-ffi must be usable."""
    jit = importlib.import_module("sglang.kernels.jit")
    assert jit is not None

    version = _installed_tvm_ffi_version()
    assert version is not None, "apache-tvm-ffi is not installed"
    assert _platform_stubs._parse_tvm_ffi_version(
        version
    ) >= _platform_stubs._parse_tvm_ffi_version(MINIMUM_TVM_FFI), (
        f"apache-tvm-ffi {version} is older than {MINIMUM_TVM_FFI}"
    )

    tvm_ffi = importlib.import_module("tvm_ffi")
    assert tvm_ffi is not None


def test_parse_tvm_ffi_version_handles_post_suffix():
    """0.1.15.post2 style versions must parse and order correctly."""
    parse = _platform_stubs._parse_tvm_ffi_version

    assert parse("0.1.15.post2") == ((0, 1, 15), 2)
    assert parse("0.1.15") == ((0, 1, 15), 0)
    assert parse("0.1.15.post1") < parse("0.1.15.post2")
    assert parse("0.1.11") < parse("0.1.15.post2")
    assert parse("v0.1.15.post2") == parse("0.1.15.post2")

    with pytest.raises(ValueError):
        parse("not-a-version")


def test_require_tvm_ffi_version_accepts_installed_version():
    version = _installed_tvm_ffi_version()
    if version is None:
        pytest.skip("apache-tvm-ffi is not installed")
    assert _platform_stubs.require_tvm_ffi_version(MINIMUM_TVM_FFI) == version


def test_require_tvm_ffi_version_missing_raises(monkeypatch):
    """A missing install must raise a clear error, not silently degrade."""
    monkeypatch.setattr(_platform_stubs, "_tvm_ffi_version", lambda: None)
    with pytest.raises(RuntimeError, match="not installed"):
        _platform_stubs.require_tvm_ffi_version(MINIMUM_TVM_FFI)


def test_require_tvm_ffi_version_too_old_raises(monkeypatch):
    """An old install (e.g. 0.1.11) must raise a clear error."""
    monkeypatch.setattr(_platform_stubs, "_tvm_ffi_version", lambda: "0.1.11")
    with pytest.raises(RuntimeError, match="older than the required"):
        _platform_stubs.require_tvm_ffi_version(MINIMUM_TVM_FFI)
