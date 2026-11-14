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
