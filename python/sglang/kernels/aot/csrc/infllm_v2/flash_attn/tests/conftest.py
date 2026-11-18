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


# NOTE: this conftest deliberately does NOT install a meta_path finder that
# raises ImportError for ``sglang`` / ``torch`` / ``numpy``. Such a finder
# would also break pytest's own plugin loading and rootdir detection in CI
# (pytest imports its plugins through the normal import machinery), which is
# what caused the previous environment-preparation failures. The tests in
# this directory are pure standard library and simply never import sglang or
# torch, so no hard block is needed.
#
# Keep the tests stdlib-only by construction: do not add third-party imports
# to the test modules in this directory.


def pytest_configure(config):  # pragma: no cover - only runs under pytest
    """Pin the pytest rootdir to this directory when pytest is present.

    Without this, pytest can walk up to the repository root, import the
    ``sglang`` package during collection and trigger the full dependency
    install. The sibling ``pytest.ini`` already sets ``rootdir``/``testpaths``;
    this hook makes the intent explicit and keeps the stdlib-only runner path
    (which never calls this hook) unaffected.
    """
    if pytest is None:
        return
    try:
        config.inifile = os.path.join(_HERE, "pytest.ini")
        config.rootdir = _HERE
    except Exception:
        # Never let a pytest-internal attribute change break collection.
        pass
