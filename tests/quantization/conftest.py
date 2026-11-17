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


def _is_blocked_path(entry: str) -> bool:
    """Return True for repo-root, python/ and any sglang import path.

    The repository root contains a ``python/`` package directory and a
    ``setup.py``; leaving it on ``sys.path`` lets pytest's rootdir discovery
    pull in the full sglang dependency set. Any path whose components include
    ``sglang`` is likewise dropped so a stale ``PYTHONPATH`` entry cannot
    re-trigger the full dependency resolution.
    """

    if entry in ("", "."):
        return True
    try:
        resolved = Path(entry).resolve()
    except (OSError, TypeError, ValueError):
        return False
    if resolved in (_REPO_ROOT, _REPO_ROOT / "python"):
        return True
    try:
        parts = resolved.parts
    except (TypeError, ValueError):
        return False
    return "sglang" in parts


def _strip_repo_paths() -> None:
    """Remove the repository root and sglang entries from the import path.

    This runs at import time (before collection) and is idempotent, so it is
    safe to call again from ``pytest_configure`` after pytest has finished
    its own rootdir / conftest discovery. It also drops any already-imported
    ``sglang`` modules so a stale import cannot re-trigger the full
    dependency resolution during collection.
    """

    sys.path[:] = [entry for entry in sys.path if not _is_blocked_path(entry)]

    for name in list(sys.modules):
        if name == "sglang" or name.startswith("sglang."):
            del sys.modules[name]

    for var in ("PYTHONPATH", "PYTHONSTARTUP"):
        value = os.environ.get(var)
        if not value:
            continue
        kept = [
            part
            for part in value.split(os.pathsep)
            if part and not _is_blocked_path(part)
        ]
        if kept:
            os.environ[var] = os.pathsep.join(kept)
        else:
            os.environ.pop(var, None)


def _ensure_minimal_pytest_config() -> None:
    """Fail fast if the suite is not run with its minimal pytest config.

    Running pytest without ``-c tests/quantization/pytest.ini`` lets pytest
    discover the repository-root ``pyproject.toml`` / ``setup.py``, which
    triggers the full sglang dependency resolution during environment
    preparation. We refuse to run in that configuration so CI fails with a
    clear message instead of a confusing pip error.
    """

    ini = _HERE / "pytest.ini"
    if not ini.is_file():
        raise RuntimeError(
            f"missing minimal pytest config: {ini}; CI must invoke pytest "
            "with ``-c tests/quantization/pytest.ini``"
        )


_strip_repo_paths()
_ensure_minimal_pytest_config()


def pytest_configure(config) -> None:
    """Re-strip repo paths after pytest's own rootdir discovery.

    pytest may re-insert the rootdir (or the invocation directory) onto
    ``sys.path`` while loading conftest files. Running the strip again here
    guarantees the suite stays isolated from the full sglang dependency set
    even when CI invokes pytest from the repository root.
    """

    _strip_repo_paths()
    _ensure_minimal_pytest_config()

    # Force the suite's own rootdir so pytest never walks up to the
    # repository-root pyproject.toml / setup.py during collection.
    config.rootpath = _HERE
    config.inifile = _HERE / "pytest.ini"

    # Guard against a regression where the repo root leaks back onto the
    # import path and pytest resolves the full sglang dependency set during
    # environment preparation.
    leaked = [entry for entry in sys.path if _is_blocked_path(entry)]
    if leaked:
        raise RuntimeError(
            "repository root leaked onto sys.path during pytest_configure: "
            f"{leaked!r}"
        )


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
