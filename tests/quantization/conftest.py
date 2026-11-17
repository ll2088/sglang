# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) vLLM project
"""Minimal, self-contained pytest configuration for the activation-fallback suite.

This suite must run with only ``pytest`` installed. Importing ``sglang`` (or
any of its submodules) resolves the full runtime dependency set, which is what
made CI environment preparation fail. To keep that from happening:

* the repository root and its ``python/`` directory are stripped from
  ``sys.path`` at import time (before pytest collects anything),
* ``PYTHONPATH`` is sanitized the same way,
* the module under test is loaded directly from its source file instead of
  being imported through the ``sglang`` package.

Everything here runs at module import time so it takes effect before any test
module is collected.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
# This suite lives at <repo_root>/tests/quantization, so the repository root
# is two levels up. The sglang package itself lives under <repo_root>/python.
_REPO_ROOT = _HERE.parent.parent
_BLOCKED_PATHS = {_REPO_ROOT, _REPO_ROOT / "python"}


def _is_blocked(entry: str) -> bool:
    if not entry:
        return False
    try:
        resolved = Path(entry).resolve()
    except (OSError, TypeError, ValueError):
        return False
    if resolved in _BLOCKED_PATHS:
        return True
    # Any path that would make the ``sglang`` package importable.
    return "sglang" in resolved.parts


def _strip_paths() -> None:
    sys.path[:] = [entry for entry in sys.path if not _is_blocked(entry)]

    pythonpath = os.environ.get("PYTHONPATH", "")
    if pythonpath:
        kept = [
            part
            for part in pythonpath.split(os.pathsep)
            if part and not _is_blocked(part)
        ]
        if kept:
            os.environ["PYTHONPATH"] = os.pathsep.join(kept)
        else:
            os.environ.pop("PYTHONPATH", None)


# Strip before anything else can import the repository packages.
_strip_paths()


def _load_activation_fallback():
    """Load ``activation_fallback.py`` from its file path.

    The module only uses the standard library, so it can be loaded without
    importing the ``sglang`` package (which would pull in the full dependency
    set).
    """

    module_path = (
        _REPO_ROOT
        / "python"
        / "sglang"
        / "srt"
        / "layers"
        / "quantization"
        / "activation_fallback.py"
    )
    if not module_path.is_file():
        raise FileNotFoundError(
            f"activation_fallback.py not found at {module_path}; the suite "
            "must be run from a checkout that contains it."
        )

    spec = importlib.util.spec_from_file_location(
        "activation_fallback", module_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load activation_fallback from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ACTIVATION_FALLBACK = _load_activation_fallback()


def pytest_configure(config) -> None:
    # Re-strip in case a plugin or the rootdir discovery re-added entries.
    _strip_paths()


def pytest_collection_modifyitems(config, items) -> None:
    # Guard against accidental sglang imports during collection: if any test
    # module pulled in the sglang package, fail loudly instead of silently
    # resolving the full dependency set.
    if "sglang" in sys.modules:
        raise RuntimeError(
            "tests/quantization must not import sglang; found it in "
            "sys.modules during collection. The suite is loaded from file "
            "paths via conftest.py so CI only needs pytest installed."
        )


def pytest_report_header(config) -> str:
    return "activation-fallback suite: self-contained (pytest only)"


import pytest  # noqa: E402  (imported after path stripping on purpose)


@pytest.fixture(scope="session")
def activation_fallback():
    """The module under test, loaded from its source file."""

    return _ACTIVATION_FALLBACK
