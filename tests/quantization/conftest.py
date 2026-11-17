# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) vLLM project
"""Self-contained fixtures for the activation-fallback suite.

This conftest deliberately avoids importing ``sglang`` (or any of its
submodules). The module under test only uses the standard library, so the
suite can run with nothing but pytest installed. CI must therefore install
``tests/quantization/requirements.txt`` instead of ``pip install -e python``,
which would resolve the full sglang dependency set and fail during
environment preparation.

To keep the suite isolated we:

* strip the repository root and any ``sglang`` entries from ``sys.path`` and
  ``PYTHONPATH`` before collection, and
* load ``activation_fallback.py`` directly from its file path via
  ``importlib.util.spec_from_file_location`` so no package import is needed.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
_MODULE_PATH = (
    _REPO_ROOT
    / "python"
    / "sglang"
    / "srt"
    / "layers"
    / "quantization"
    / "activation_fallback.py"
)


def _strip_repo_paths() -> None:
    """Remove the repository root and sglang entries from the import path.

    The repository root contains a ``python/`` package directory and a
    ``setup.py``; leaving it on ``sys.path`` lets pytest's rootdir discovery
    pull in the full sglang dependency set. We also drop any ``sglang``
    entries that may have been injected via ``PYTHONPATH``.
    """

    repo_root = str(_REPO_ROOT)
    python_dir = str(_REPO_ROOT / "python")
    blocked = {repo_root, python_dir, "", "."}

    sys.path[:] = [
        entry
        for entry in sys.path
        if entry not in blocked and "sglang" not in Path(entry).parts
    ]

    for var in ("PYTHONPATH", "PYTHONSTARTUP"):
        value = os.environ.get(var)
        if not value:
            continue
        kept = [
            part
            for part in value.split(os.pathsep)
            if part and part not in blocked and "sglang" not in Path(part).parts
        ]
        if kept:
            os.environ[var] = os.pathsep.join(kept)
        else:
            os.environ.pop(var, None)


_strip_repo_paths()


def _load_activation_fallback():
    """Load ``activation_fallback.py`` from its file path.

    Loading by path keeps the module out of the ``sglang`` package namespace
    and guarantees that importing it never triggers the full dependency set.
    """

    if not _MODULE_PATH.is_file():
        raise FileNotFoundError(
            f"activation_fallback.py not found at {_MODULE_PATH}; the suite "
            "must be run from a checkout that contains the module."
        )

    spec = importlib.util.spec_from_file_location(
        "activation_fallback", _MODULE_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load activation_fallback from {_MODULE_PATH}")

    module = importlib.util.module_from_spec(spec)
    # Register under a non-sglang name so the self-containment guards in the
    # test module can assert that no ``sglang`` module was imported.
    sys.modules.setdefault("activation_fallback", module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def activation_fallback():
    """Return the activation-fallback module loaded from its source file."""

    return _load_activation_fallback()
