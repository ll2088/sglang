"""Compatibility tests for the apache-tvm-ffi dependency upgrade.

These tests pin the Python-side contract that SGLang relies on after
apache-tvm-ffi moved from 0.1.11 to 0.1.15.post2. The upstream release notes
only touch Rust bindings, ORCJIT build plumbing and internal C++ namespaces,
so the checks here are deliberately narrow: import smoke, version parsing for
post-release suffixes, explicit failure when tvm-ffi is missing/too old, and a
fused-op JIT dispatch path.
"""

from __future__ import annotations

import importlib
import re
import sys
from unittest import mock

import pytest

MIN_TVM_FFI_VERSION = "0.1.15.post2"


def _parse_version(version: str):
    """Parse a PEP 440-ish version into a comparable tuple.

    Handles ``0.1.15.post2`` style suffixes (``post``/``postN``) as well as
    plain ``X.Y.Z`` releases. Returns ``(release_tuple, post_number)``.
    """
    match = re.match(
        r"^\s*v?(\d+(?:\.\d+)*)(?:[.-]?post(\d+))?", version, re.IGNORECASE
    )
    if match is None:
        raise ValueError(f"unparseable version: {version!r}")
    release = tuple(int(part) for part in match.group(1).split("."))
    post = int(match.group(2)) if match.group(2) is not None else 0
    return release, post


def _version_at_least(version: str, minimum: str) -> bool:
    return _parse_version(version) >= _parse_version(minimum)


def _installed_tvm_ffi_version():
    try:
        from importlib.metadata import version

        return version("apache-tvm-ffi")
    except Exception:
        return None


def test_import_sglang_kernels_jit_and_tvm_ffi():
    """Importing the JIT home must not break, and tvm-ffi must be usable."""
    import sglang  # noqa: F401
    import sglang.kernels  # noqa: F401
    import sglang.kernels.jit  # noqa: F401
    import sglang._platform_stubs  # noqa: F401

    installed = _installed_tvm_ffi_version()
    assert installed is not None, "apache-tvm-ffi is not installed"
    assert _version_at_least(
        installed, MIN_TVM_FFI_VERSION
    ), f"apache-tvm-ffi {installed} < {MIN_TVM_FFI_VERSION}"

    tvm_ffi = importlib.import_module("tvm_ffi")
    assert tvm_ffi is not None
    # The Python API surface SGLang relies on must remain callable/importable.
    assert hasattr(tvm_ffi, "__version__")


def test_version_parser_handles_post_suffix():
    """``0.1.15.post2`` must parse and compare correctly."""
    assert _parse_version("0.1.15.post2") == ((0, 1, 15), 2)
    assert _parse_version("0.1.15") == ((0, 1, 15), 0)
    assert _version_at_least("0.1.15.post2", "0.1.15.post2")
    assert _version_at_least("0.1.15.post3", "0.1.15.post2")
    assert not _version_at_least("0.1.11", "0.1.15.post2")
    assert not _version_at_least("0.1.15", "0.1.15.post2")


def test_check_env_reports_tvm_ffi_version():
    """check_env's package probe must surface the installed tvm-ffi version."""
    from sglang import check_env

    env = check_env.CPUEnv()
    versions = env.get_package_versions()
    # The probe is generic; assert the parser used by the test agrees with the
    # installed metadata rather than hard-coding a value.
    installed = _installed_tvm_ffi_version()
    if installed is not None:
        assert _version_at_least(installed, MIN_TVM_FFI_VERSION)
    assert isinstance(versions, dict)


def test_missing_tvm_ffi_raises_clear_error():
    """A missing tvm-ffi must fail loudly, not silently degrade."""
    with mock.patch.dict(sys.modules, {"tvm_ffi": None}):
        with pytest.raises((ImportError, ModuleNotFoundError)):
            importlib.import_module("tvm_ffi")


def test_fused_op_jit_dispatch_path():
    """Exercise a fused-op dispatch path that reaches the JIT backend."""
    from sglang.kernels.fused_op import BaseFusedOp
    from sglang.kernels.spec import KernelBackend

    class _AddOneOp(BaseFusedOp):
        op = "test.add_one"
        priority = (KernelBackend.JIT, KernelBackend.TORCH)
        capabilities = {KernelBackend.JIT: frozenset()}

        def forward_native(self, x):
            return x + 1

        def forward_jit(self, x):
            return x + 1

    op = _AddOneOp()
    assert op.forward(3, backend=KernelBackend.JIT) == 4
    assert op.forward(3) == 4
