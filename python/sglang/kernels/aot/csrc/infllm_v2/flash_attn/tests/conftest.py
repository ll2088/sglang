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
