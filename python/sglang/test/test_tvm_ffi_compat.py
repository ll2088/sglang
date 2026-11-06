"""Compatibility tests for the apache-tvm-ffi dependency bump.

These tests are intentionally dependency-light: they only use the standard
library plus torch (already a hard dependency of sglang), so they can run in
the same environment as the rest of the suite without pulling in a kernel
backend or triggering JIT compilation.
"""

import importlib
import importlib.metadata
import re
import sys

import pytest


MIN_TVM_FFI_VERSION = "0.1.15.post2"


def _parse_version(version: str):
    """Parse a PEP 440-ish version into a comparable tuple.

    Handles ``post`` suffixes such as ``0.1.15.post2`` as well as the
    ``0.1.15-post2`` spelling used in some release notes.
    """
    match = re.match(
        r"^\s*v?(\d+(?:\.\d+)*)(?:[.-]?(?:post|rev|r)(\d+))?", version
    )
    if match is None:
        raise ValueError(f"unparsable version: {version!r}")
    release = tuple(int(part) for part in match.group(1).split("."))
    post = int(match.group(2)) if match.group(2) is not None else 0
    return release, post


def _version_at_least(version: str, minimum: str) -> bool:
    return _parse_version(version) >= _parse_version(minimum)


def _tvm_ffi_version():
    try:
        return importlib.metadata.version("apache-tvm-ffi")
    except importlib.metadata.PackageNotFoundError:
        return None


def test_parse_post_version():
    assert _parse_version("0.1.15.post2") == ((0, 1, 15), 2)
    assert _parse_version("0.1.15-post2") == ((0, 1, 15), 2)
    assert _parse_version("0.1.11") == ((0, 1, 11), 0)
    assert _version_at_least("0.1.15.post2", MIN_TVM_FFI_VERSION)
    assert not _version_at_least("0.1.11", MIN_TVM_FFI_VERSION)


def test_tvm_ffi_installed_version():
    version = _tvm_ffi_version()
    if version is None:
        pytest.skip("apache-tvm-ffi is not installed in this environment")
    assert _version_at_least(version, MIN_TVM_FFI_VERSION), (
        f"apache-tvm-ffi {version} is older than the required "
        f"{MIN_TVM_FFI_VERSION}"
    )


def test_import_kernels_jit_after_tvm_ffi():
    # Importing the kernels namespace must not require a kernel backend or
    # trigger JIT compilation; it is metadata-only.
    import sglang.kernels  # noqa: F401
    import sglang.kernels.jit  # noqa: F401

    version = _tvm_ffi_version()
    if version is not None:
        assert _version_at_least(version, MIN_TVM_FFI_VERSION)


def test_platform_stubs_import_is_safe():
    # _platform_stubs must be importable on any platform; installation is
    # gated on darwin/arm64 + MPS availability.
    module = importlib.import_module("sglang._platform_stubs")
    assert hasattr(module, "install_platform_stubs")
    # Calling it on a non-MPS platform is a no-op and must not raise.
    module.install_platform_stubs()


def test_platform_stubs_reports_missing_tvm_ffi(monkeypatch):
    """A missing/too-old tvm-ffi must surface a clear error, not silently
    degrade, when the compatibility helper is asked to validate it."""
    from sglang._platform_stubs import require_tvm_ffi_version

    monkeypatch.setattr(
        "sglang._platform_stubs._tvm_ffi_version", lambda: None, raising=False
    )
    with pytest.raises(RuntimeError, match="apache-tvm-ffi"):
        require_tvm_ffi_version(MIN_TVM_FFI_VERSION)

    monkeypatch.setattr(
        "sglang._platform_stubs._tvm_ffi_version", lambda: "0.1.11", raising=False
    )
    with pytest.raises(RuntimeError, match="0.1.11"):
        require_tvm_ffi_version(MIN_TVM_FFI_VERSION)


def test_fused_op_native_path():
    """Exercise the pure-torch dispatch path of BaseFusedOp.

    This covers the fused_op code path without requiring any kernel backend
    or JIT compilation, and is the path most likely to be affected by a
    dependency bump in the surrounding runtime.
    """
    import torch

    from sglang.kernels.fused_op import BaseFusedOp
    from sglang.kernels.spec import KernelBackend

    class _AddOne(BaseFusedOp):
        op = "test.add_one"

        def forward_native(self, x):
            return x + 1

    op = _AddOne()
    x = torch.zeros(4)
    out = op(x)
    assert torch.equal(out, torch.ones(4))
    # Explicit backend selection must also work for the native path.
    out_native = op(x, backend=KernelBackend.TORCH)
    assert torch.equal(out_native, torch.ones(4))
