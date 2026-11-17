# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) vLLM project
"""Self-contained conftest for the activation-fallback test suite.

This suite must run without importing the ``sglang`` package (no GPU, no
compiled kernels, no full dependency set). The module under test only uses the
standard library, so it is loaded directly from its file path by the tests.

Keeping this conftest free of ``sglang`` imports is what lets CI install only
``tests/quantization/requirements.txt`` instead of the full package.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

# The module under test only uses the standard library, so it is loaded
# directly from its file path. This keeps the suite runnable with only pytest
# installed and avoids importing the ``sglang`` package (which would pull in
# the full dependency set and fail in minimal CI environments).
#
# The path is resolved relative to this conftest file so it works regardless
# of the pytest rootdir / importmode chosen by CI. We deliberately do NOT
# import ``sglang`` anywhere in this file: any such import would trigger the
# full dependency resolution that this suite is designed to avoid.
#
# Guard against a parent conftest.py / pyproject.toml being collected: if the
# repository root ends up on sys.path, ``import sglang`` can be attempted
# during collection and resolve the full dependency set. We therefore resolve
# the module path from this file only and never rely on sys.path.
#
# We also defensively drop the repository root from ``sys.path`` (if present)
# so that a stray ``import sglang`` anywhere in the collection chain cannot
# resolve the top-level package and pull in its full dependency set.
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[1]

# Drop every path that could make ``import sglang`` resolve the top-level
# package (whose ``__init__`` pulls in the full dependency set). This covers
# the repository root, the ``python/`` subdirectory, and this test directory
# itself, regardless of how pytest was invoked or what an editable install
# injected.
_FORBIDDEN_PATHS = {
    str(_REPO_ROOT),
    str(_REPO_ROOT / "python"),
    str(_THIS_DIR),
}
sys.path[:] = [p for p in sys.path if p not in _FORBIDDEN_PATHS]

# Keep the suite hermetic: never let a stray ``PYTHONPATH`` re-introduce the
# repository root during collection.
os.environ.pop("PYTHONPATH", None)
_ACTIVATION_FALLBACK_PATH = (
    _REPO_ROOT
    / "python"
    / "sglang"
    / "srt"
    / "layers"
    / "quantization"
    / "activation_fallback.py"
)

# Fail fast with a clear message if the repository layout is unexpected, so CI
# reports a layout problem instead of a confusing import error.
if not _ACTIVATION_FALLBACK_PATH.is_file():
    raise FileNotFoundError(
        "activation_fallback.py not found at "
        f"{_ACTIVATION_FALLBACK_PATH}; the tests/quantization suite must be "
        "run from a checkout of the repository."
    )


def _load_activation_fallback():
    """Load ``activation_fallback`` from its file path without importing sglang.

    The module is loaded under a private name that does not start with
    ``sglang`` so that no import machinery ever tries to resolve the
    ``sglang`` package (whose ``__init__`` pulls in the full dependency set).
    """

    module_name = "activation_fallback_under_test"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing

    if not _ACTIVATION_FALLBACK_PATH.is_file():
        raise FileNotFoundError(
            f"activation_fallback.py not found at {_ACTIVATION_FALLBACK_PATH}"
        )

    spec = importlib.util.spec_from_file_location(
        module_name, _ACTIVATION_FALLBACK_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(
            f"could not create import spec for {_ACTIVATION_FALLBACK_PATH}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _assert_no_sglang_imported() -> None:
    """Fail fast if the ``sglang`` package was imported during collection.

    Importing ``sglang`` triggers the full dependency resolution (py-spy,
    quack-kernels, pyzmq, ...) which is exactly what this suite avoids. If it
    ever happens, surface a clear error instead of a confusing dependency
    failure.
    """

    sglang_modules = [
        name
        for name in sys.modules
        if name == "sglang" or name.startswith("sglang.")
    ]
    if sglang_modules:
        raise RuntimeError(
            "tests/quantization must not import the sglang package; the suite "
            "is loaded from source files via conftest.py so CI only needs "
            "pytest installed. Found: "
            f"{sglang_modules}. Check for a stray ``import sglang`` in the "
            "collection chain or a parent conftest.py."
        )


# Expose the loaded module as a fixture so tests do not need to import sglang.
import pytest  # noqa: E402  (imported after the loader helpers for clarity)


@pytest.fixture(scope="session")
def activation_fallback():
    """Return the activation_fallback module loaded from its source file."""

    _assert_no_sglang_imported()
    return _load_activation_fallback()
