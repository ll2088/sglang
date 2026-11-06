"""Compatibility tests for the apache-tvm-ffi 0.1.15.post2 upgrade.

These tests pin the behaviour the Python side relies on after the patch
upgrade: the installed version is at least 0.1.15.post2, the post-release
version string parses correctly, a missing/too-old install fails loudly in
``sglang._platform_stubs``, and the fused-op dispatch path still works.
"""

import importlib
import importlib.metadata
import sys

import pytest

from sglang._platform_stubs import (
    _parse_tvm_ffi_version,
    require_tvm_ffi_version,
)

MINIMUM_TVM_FFI = "0.1.15.post2"


def _installed_tvm_ffi_version():
    try:
        return importlib.metadata.version("apache-tvm-ffi")
    except importlib.metadata.PackageNotFoundError:
        return None


def test_installed_tvm_ffi_meets_minimum():
    version = _installed_tvm_ffi_version()
    if version is None:
        pytest.skip("apache-tvm-ffi is not installed in this environment")
    assert _parse_tvm_ffi_version(version) >= _parse_tvm_ffi_version(
        MINIMUM_TVM_FFI
    )


def test_parse_post_release_version():
    assert _parse_tvm_ffi_version("0.1.15.post2") == ((0, 1, 15), 2)
    assert _parse_tvm_ffi_version("0.1.15") == ((0, 1, 15), 0)
    assert _parse_tvm_ffi_version("0.1.11") < _parse_tvm_ffi_version(
        MINIMUM_TVM_FFI
    )
    assert _parse_tvm_ffi_version("0.1.15.post2") > _parse_tvm_ffi_version(
        "0.1.15.post1"
    )


def test_parse_rejects_garbage():
    with pytest.raises(ValueError):
        _parse_tvm_ffi_version("not-a-version")


def test_require_tvm_ffi_version_reports_installed():
    version = _installed_tvm_ffi_version()
    if version is None:
        pytest.skip("apache-tvm-ffi is not installed in this environment")
    assert require_tvm_ffi_version(MINIMUM_TVM_FFI) == version


def test_require_tvm_ffi_version_missing(monkeypatch):
    import sglang._platform_stubs as stubs

    monkeypatch.setattr(stubs, "_tvm_ffi_version", lambda: None)
    with pytest.raises(RuntimeError, match="not installed"):
        stubs.require_tvm_ffi_version(MINIMUM_TVM_FFI)


def test_require_tvm_ffi_version_too_old(monkeypatch):
    import sglang._platform_stubs as stubs

    monkeypatch.setattr(stubs, "_tvm_ffi_version", lambda: "0.1.11")
    with pytest.raises(RuntimeError, match="older than"):
        stubs.require_tvm_ffi_version(MINIMUM_TVM_FFI)


def test_import_kernels_jit():
    module = importlib.import_module("sglang.kernels.jit")
    assert module is not None
    assert "sglang.kernels.jit" in sys.modules


def test_fused_op_native_dispatch():
    import torch

    from sglang.kernels.fused_op import BaseFusedOp

    class _AddOneOp(BaseFusedOp):
        op = "test.add_one"

        def forward_native(self, x):
            return x + 1

    op = _AddOneOp()
    x = torch.zeros(4)
    out = op(x)
    assert torch.equal(out, torch.ones(4))
    assert op.available_backends()
