# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Self-contained conftest for tests/quantization.

This directory's tests must run without a full ``sglang`` install. The
repository-level conftest.py imports ``sglang`` (and therefore its full
dependency set), so we deliberately keep this conftest free of any sglang
imports and only add the source tree to ``sys.path`` so tests can locate the
module under test by file path.

If a future test in this directory needs sglang, it should be moved out of
this directory rather than adding an import here.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _find_repo_root() -> Path | None:
    """Walk upwards from this file to locate the repository root.

    The repository root is identified by the presence of the
    ``python/sglang/srt/layers/quantization`` directory. Returns ``None`` when
    the source tree is not present (e.g. a partial checkout); tests that need
    it will skip with an explicit message instead of failing during collection.
    """

    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (
            candidate
            / "python"
            / "sglang"
            / "srt"
            / "layers"
            / "quantization"
        ).is_dir():
            return candidate
    return None


_REPO_ROOT = _find_repo_root()

if _REPO_ROOT is not None:
    # Make the source tree importable by file path without requiring an
    # editable install. This is a no-op for tests that load modules directly
    # from their file path, but keeps the door open for future tests that
    # import sibling modules by package path.
    _python_dir = str(_REPO_ROOT / "python")
    if _python_dir not in sys.path:
        sys.path.insert(0, _python_dir)
