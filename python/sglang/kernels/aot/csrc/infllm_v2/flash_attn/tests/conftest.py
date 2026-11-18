"""Minimal conftest for the int64 accumulator tests.

This file exists so that pytest can collect the tests in this directory
without importing the ``sglang`` package (or any of its heavy third-party
dependencies such as torch / numpy). The tests here are pure standard
library and must stay runnable in environments where the full sglang
dependency set cannot be installed.

Do not add third-party imports to this file.
"""

import os
import sys

# Make sure the tests directory itself is importable so that
# ``import test_int64_accum_sizes`` works regardless of the pytest rootdir.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# This conftest must stay importable even when pytest is not installed (e.g.
# when the tests are executed through the stdlib-only runner
# ``run_int64_accum_tests.py``). Guard the pytest import so that a missing
# pytest never turns into a collection-time ImportError.
try:
    import pytest  # noqa: F401
except ImportError:  # pragma: no cover - exercised only without pytest
    pytest = None

# When pytest is unavailable (e.g. the stdlib-only runner path), make sure the
# tests directory is still importable and that no pytest-only hooks are
# required. This keeps the int64 accumulator checks runnable in a bare Python
# environment without installing the heavy sglang dependency set.
if pytest is None:  # pragma: no cover - exercised only without pytest
    _HERE = os.path.dirname(os.path.abspath(__file__))
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)

# Keep the tests stdlib-only: make sure the repository root (which contains the
# ``sglang`` package) is not on ``sys.path`` while these tests are collected.
# Otherwise pytest's rootdir detection can import ``sglang`` and trigger the
# full dependency install during environment preparation.
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, os.pardir, os.pardir, os.pardir, os.pardir, os.pardir))
for _entry in list(sys.path):
    if not _entry:
        continue
    try:
        _resolved = os.path.normpath(os.path.abspath(_entry))
    except Exception:
        continue
    if _resolved == _REPO_ROOT:
        sys.path.remove(_entry)


# Hard block: install a meta_path finder that raises ImportError for the
# ``sglang`` package (and its submodules) before pytest imports any test
# module. This prevents pytest plugins or rootdir detection from triggering
# the heavy sglang dependency resolution during collection.
#
# The blocked set also covers the heavy third-party packages that the sglang
# package pulls in (torch / numpy / etc.). Blocking them here means that even
# if some plugin or rootdir detection tries to import them, collection will
# fail fast with a clear ImportError instead of attempting a network install
# of the full sglang dependency set during CI environment preparation.
class _BlockSglangImport:
    """Meta path finder that refuses to import ``sglang`` and its heavy deps."""

    _BLOCKED = ("sglang", "torch", "numpy", "cute", "cutlass")

    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".", 1)[0]
        if root in self._BLOCKED:
            raise ImportError(
                "Importing '{}' is blocked by the int64 accumulator test "
                "conftest to keep CI environment preparation lightweight.".format(fullname)
            )
        return None


if not any(isinstance(f, _BlockSglangImport) for f in sys.meta_path):
    sys.meta_path.insert(0, _BlockSglangImport())

# Also drop any already-imported heavy modules from sys.modules so that a
# partially imported sglang/torch (e.g. from a previous collection attempt)
# cannot leak into this test session.
for _mod in list(sys.modules):
    _root = _mod.split(".", 1)[0]
    if _root in _BlockSglangImport._BLOCKED:
        del sys.modules[_mod]
