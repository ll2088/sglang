# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) vLLM project
"""Minimal conftest for the activation-fallback suite.

This suite must run with only pytest installed: it must never import the
``sglang`` package (which resolves the full runtime dependency set and is what
made CI environment preparation fail). To keep that guarantee we

* strip the repository root and its ``python/`` directory from ``sys.path``
  (and from ``PYTHONPATH``) at import time and again in ``pytest_configure``,
  so pytest's rootdir discovery cannot resolve ``sglang``, and
* load ``activation_fallback.py`` directly from its source file instead of
  importing it through the ``sglang`` package.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
_BLOCKED_PATHS = {_REPO_ROOT, _REPO_ROOT / "python"}


def _strip_repo_root_from_sys_path() -> None:
    """Remove repo-root / python-dir / sglang entries from ``sys.path``."""

    cleaned = []
    for entry in sys.path:
        if not entry:
            cleaned.append(entry)
            continue
        try:
            resolved = Path(entry).resolve()
        except (OSError, TypeError, ValueError):
            cleaned.append(entry)
            continue
        if resolved in _BLOCKED_PATHS or "sglang" in resolved.parts:
            continue
        cleaned.append(entry)
    sys.path[:] = cleaned


def _strip_repo_root_from_pythonpath() -> None:
    """Remove repo-root / python-dir / sglang entries from ``PYTHONPATH``."""

    value = os.environ.get("PYTHONPATH", "")
    if not value:
        return
    kept = []
    for part in value.split(os.pathsep):
        if not part:
            continue
        try:
            resolved = Path(part).resolve()
        except (OSError, TypeError, ValueError):
            kept.append(part)
            continue
        if resolved in _BLOCKED_PATHS or "sglang" in resolved.parts:
            continue
        kept.append(part)
    if kept:
        os.environ["PYTHONPATH"] = os.pathsep.join(kept)
    else:
        os.environ.pop("PYTHONPATH", None)


_strip_repo_root_from_sys_path()
_strip_repo_root_from_pythonpath()


def pytest_configure(config: pytest.Config) -> None:
    # pytest may re-insert the rootdir during collection; strip it again.
    _strip_repo_root_from_sys_path()
    _strip_repo_root_from_pythonpath()


def pytest_collection_modifyitems(config: pytest.Config, items) -> None:
    # Collection may re-insert the rootdir; strip it once more before the
    # tests run so no sglang import can resolve through sys.path.
    _strip_repo_root_from_sys_path()
    _strip_repo_root_from_pythonpath()


def _load_activation_fallback():
    """Load ``activation_fallback.py`` from its source file.

    The module is intentionally importable with only the standard library, so
    loading it by path keeps the suite free of any ``sglang`` import.
    """

    module_path = _REPO_ROOT / "python" / "sglang" / "srt" / "layers" / "quantization" / "activation_fallback.py"
    if not module_path.is_file():
        pytest.skip(f"activation_fallback.py not found at {module_path}")

    spec = importlib.util.spec_from_file_location(
        "activation_fallback", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def activation_fallback():
    """The module under test, loaded from its source file (no sglang import)."""

    return _load_activation_fallback()
