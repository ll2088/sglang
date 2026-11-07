"""Minimal conftest for the kernel tests that must run without sglang.

These tests only need numpy and pytest.  Keeping this file free of torch and
sglang imports lets CI run them in a small dedicated job instead of installing
the full runtime, which is what previously failed during environment setup.
"""

import sys
from pathlib import Path

import pytest


# Make the repository root importable so tests can load pure-Python helper
# modules by explicit path without installing the sglang package.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _has_numpy() -> bool:
    try:
        import numpy  # noqa: F401
    except Exception:  # pragma: no cover - depends on the environment
        return False
    return True


_HAS_NUMPY = _has_numpy()


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_sglang: test needs the sglang package to be importable",
    )
    config.addinivalue_line(
        "markers",
        "requires_numpy: test needs numpy to be importable",
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests marked ``requires_sglang`` when sglang is not importable.

    The BF16 capability tests are pure Python and must run in a minimal
    pytest-only job; anything that genuinely needs the runtime is skipped
    instead of failing collection.
    """
    try:
        import sglang  # noqa: F401
    except Exception:  # pragma: no cover - depends on the environment
        skip = pytest.mark.skip(reason="sglang is not importable")
        for item in items:
            if "requires_sglang" in item.keywords:
                item.add_marker(skip)

    if not _HAS_NUMPY:
        skip_np = pytest.mark.skip(reason="numpy is not importable")
        for item in items:
            if "requires_numpy" in item.keywords:
                item.add_marker(skip_np)
