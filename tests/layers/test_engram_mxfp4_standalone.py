"""Standalone entry point for the stdlib-only MXFP4 Engram self-check.

This module exists so CI can run the MXFP4 layout / E2M1 decode / row-width
allocation checks without installing the full sglang dependency chain
(torch, triton, ...). It imports the self-check module by file path and
re-exports its ``test_*`` functions so pytest collects them directly.

Run with::

    python -m pytest -q tests/layers/test_engram_mxfp4_standalone.py

No third-party dependency beyond pytest is required.
"""

from __future__ import annotations

import importlib.util
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PYTHON_DIR = os.path.join(_REPO_ROOT, "python")
if os.path.isdir(_PYTHON_DIR) and _PYTHON_DIR not in sys.path:
    sys.path.insert(0, _PYTHON_DIR)

_SELFCHECK_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "test_engram_mxfp4_selfcheck.py",
)


def _load_selfcheck():
    """Load the stdlib-only self-check module by file path.

    Loading by path avoids importing the ``tests`` package (which may pull in
    heavy dependencies) and keeps this entry point usable in a minimal CI
    environment.
    """
    if not os.path.isfile(_SELFCHECK_PATH):
        raise ImportError(f"self-check module not found at {_SELFCHECK_PATH!r}")
    spec = importlib.util.spec_from_file_location(
        "_engram_mxfp4_selfcheck_standalone", _SELFCHECK_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load self-check module from {_SELFCHECK_PATH!r}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_selfcheck = _load_selfcheck()

# Re-export every ``test_*`` callable so pytest collects them here as well.
for _name, _obj in list(vars(_selfcheck).items()):
    if _name.startswith("test_") and callable(_obj):
        globals()[_name] = _obj


def test_standalone_entry_exposes_selfcheck_cases() -> None:
    """Guard: the standalone entry must expose the full self-check suite."""
    names = [
        name
        for name, obj in vars(_selfcheck).items()
        if name.startswith("test_") and callable(obj)
    ]
    assert len(names) >= 40, f"expected >= 40 self-check cases, got {len(names)}"
    assert "test_selfcheck_suite_is_collectable" in names
