# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lightweight conftest for the self-contained activation-fallback suite.

This suite must run without installing the full sglang runtime dependency set
(no GPU, no compiled kernels). The module under test only uses the standard
library, so it is loaded directly from its file path by the test module.

This conftest deliberately does NOT import ``sglang`` and does NOT add the
repository root to ``sys.path`` in a way that would trigger the root
``conftest.py`` (which may import the full runtime). It only makes the
``python/`` source tree importable for tests that want to import the module
under test by package path, without requiring ``pip install -e python``.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _find_repo_root(start: Path) -> Path | None:
    """Walk upwards from ``start`` looking for the repository root.

    The repository root is identified by the presence of the ``python``
    directory containing the ``sglang`` package. Returns ``None`` when the
    source tree is not present (e.g. a partial checkout), in which case the
    test module skips itself with an explicit message.
    """

    for candidate in (start, *start.parents):
        if (candidate / "python" / "sglang" / "srt").is_dir():
            return candidate
    return None


_REPO_ROOT = _find_repo_root(Path(__file__).resolve().parent)

if _REPO_ROOT is not None:
    _PYTHON_DIR = _REPO_ROOT / "python"
    if _PYTHON_DIR.is_dir():
        _python_dir_str = str(_PYTHON_DIR)
        if _python_dir_str not in sys.path:
            # Append rather than prepend so an installed sglang (if any) still
            # wins; this suite does not need the package to be importable.
            sys.path.append(_python_dir_str)
