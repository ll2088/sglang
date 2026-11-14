"""Lightweight pytest configuration for the stdlib-only Engram MXFP4 tests.

CI environment preparation can fail while installing the full sglang
dependency tree (torch, triton, ...). The MXFP4 layout module and its
self-check are stdlib-only, so this conftest keeps collection working in a
minimal environment by ensuring the repository ``python`` directory is on
``sys.path`` without importing the ``sglang`` package itself.

This file intentionally imports only the standard library.
"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PYTHON_DIR = os.path.join(_REPO_ROOT, "python")

if os.path.isdir(_PYTHON_DIR) and _PYTHON_DIR not in sys.path:
    sys.path.insert(0, _PYTHON_DIR)


def pytest_addoption(parser):  # noqa: ARG001 - pytest hook signature
    """Register the stdlib-only marker used by the Engram MXFP4 tests.

    CI environment preparation can fail while installing the full sglang
    dependency tree (torch, triton, py-spy, pybase64, pydantic,
    quack-kernels, ...). The MXFP4 layout module and its self-check are
    stdlib-only, so this conftest must stay importable without that tree.
    Registering the marker here (instead of relying on the repository root
    ``pytest.ini``/``pyproject``) keeps the tests collectable when only
    pytest is installed and the root configuration is not loaded.
    """
    parser.addini(
        "engram_mxfp4_stdlib_only",
        "Run the stdlib-only Engram MXFP4 tests without the sglang tree.",
        default="true",
    )
    parser.addoption(
        "--engram-mxfp4-stdlib-only",
        action="store_true",
        default=False,
        help=(
            "Run only the stdlib-only Engram MXFP4 tests and never import "
            "the sglang package (used by the minimal CI entry point)."
        ),
    )


def _install_sglang_stub() -> None:
    """Pre-seed a stub ``sglang`` module so no import chain pulls it in.

    CI environment preparation can fail while resolving the full sglang
    dependency tree (torch, triton, py-spy, pybase64, pydantic, pyzmq,
    quack-kernels, regex, ...). The MXFP4 layout module and its self-check are
    stdlib-only and are loaded by file path, so the ``sglang`` package must
    never be imported here. Pre-seeding ``sys.modules['sglang']`` with a stub
    makes any accidental ``import sglang`` a no-op instead of triggering the
    full dependency resolution that fails during CI preparation.
    """
    if "sglang" not in sys.modules:
        import types

        sys.modules["sglang"] = types.ModuleType("sglang")


def pytest_configure(config):  # noqa: ARG001 - pytest hook signature
    """Keep the stdlib-only Engram MXFP4 tests collectable without sglang.

    The repository root ``conftest.py`` (and any ``pytest.ini``/``pyproject``
    ``addopts``) may import the ``sglang`` package, which pulls in the full
    dependency tree (torch, triton, ...). CI environment preparation can fail
    while installing that tree, so the MXFP4 layout tests must not depend on
    it. This hook ensures the source directory is importable and pre-seeds a
    ``sglang`` stub so no import chain can trigger the full dependency
    install; the layout module itself is loaded by file path in the test
    modules.
    """
    if os.path.isdir(_PYTHON_DIR) and _PYTHON_DIR not in sys.path:
        sys.path.insert(0, _PYTHON_DIR)

    # Always stub sglang for this directory: the layout module and the
    # self-check are loaded by file path, so importing the real package would
    # only re-trigger the dependency resolution this conftest exists to avoid.
    _install_sglang_stub()


def pytest_sessionstart(session):  # noqa: ARG001 - pytest hook signature
    """Stub ``sglang`` before collection so no import chain resolves it.

    CI environment preparation can fail while resolving the full sglang
    dependency tree (torch, triton, py-spy, pybase64, pydantic, pyzmq,
    quack-kernels, regex, ...). ``pytest_configure`` runs after the root
    ``conftest.py`` has already been imported, so a root-level ``import
    sglang`` could still trigger that resolution. This hook runs earlier in
    the session lifecycle and pre-seeds the stub, keeping the MXFP4 layout
    tests collectable in a minimal environment.
    """
    if os.path.isdir(_PYTHON_DIR) and _PYTHON_DIR not in sys.path:
        sys.path.insert(0, _PYTHON_DIR)
    _install_sglang_stub()


def pytest_collection_modifyitems(config, items):  # noqa: ARG001
    """Keep the stdlib-only MXFP4 tests runnable without the sglang tree.

    CI environment preparation can fail while resolving the full sglang
    dependency tree. The MXFP4 layout module and its self-check are
    stdlib-only, so the tests under ``tests/layers`` must not require that
    install. This hook is a no-op for other test directories and only
    documents the intent; the actual dependency-free execution is achieved by
    running pytest with ``PYTHONPATH=python`` and ``--confcutdir=tests/layers``
    so the repository root conftest (which may import sglang) is not loaded.
    """
    return None
