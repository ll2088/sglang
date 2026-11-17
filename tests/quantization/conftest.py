# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Self-contained conftest for the activation-fallback test suite.

This suite must run without importing the ``sglang`` package (no GPU, no
compiled kernels, no full dependency set). The module under test only uses the
standard library, so it is loaded directly from its file path by the tests.

Keeping this conftest free of ``sglang`` imports is what lets CI install only
``tests/quantization/requirements.txt`` instead of the full package.
"""

from __future__ import annotations

import importlib.util
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
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[1]
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
    """Load ``activation_fallback`` from its file path without importing sglang."""

    module_name = "sglang_activation_fallback_under_test"
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


# Expose the loaded module as a fixture so tests do not need to import sglang.
import pytest  # noqa: E402  (imported after the loader helpers for clarity)


@pytest.fixture(scope="session")
def activation_fallback():
    """Return the activation_fallback module loaded from its source file."""

    return _load_activation_fallback()
