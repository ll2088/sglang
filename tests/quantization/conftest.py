# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal conftest for the activation-fallback test suite.

This suite is intentionally self-contained: it must run without installing the
``sglang`` package (no GPU, no compiled kernels, no third-party deps). The
conftest therefore only performs path and file-existence checks and does not
import anything from ``sglang``.

CI jobs that run this directory should install only
``tests/quantization/requirements.txt`` (pytest) and must NOT run
``pip install -e python``, which pulls the full sglang runtime dependency set
and fails on GPU-less runners. The directory-local ``pytest.ini`` pins
rootdir/testpaths to this directory so the repository-root pytest
configuration is not picked up.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest

# This suite is intentionally self-contained: it must run without installing
# the ``sglang`` package. The activation-fallback tests load the module under
# test directly from its file path, so no import of ``sglang`` is required.
#
# We deliberately do NOT monkeypatch ``builtins.__import__`` here. Doing so
# would break unrelated tooling (e.g. pytest plugins, coverage, or future
# tests in this directory that legitimately need to import ``sglang`` when it
# is installed). Instead, the individual test module loads the module under
# test from its file path and fails with a precise message if the source tree
# is missing.

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Expose the ``python`` source tree so that, if a future test in this
# directory does need to import ``sglang``, it can do so from the checkout
# without a prior ``pip install -e python``. Importing ``sglang`` itself is
# still not required by the activation-fallback suite, and this conftest does
# not import it.
_PYTHON_ROOT = _REPO_ROOT / "python"
if _PYTHON_ROOT.is_dir() and str(_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYTHON_ROOT))

# Fail fast with an actionable diagnostic if the module under test is missing.
# CI logs otherwise show an opaque FileNotFoundError from the test module.
_ACTIVATION_FALLBACK_PATH = (
    _PYTHON_ROOT
    / "sglang"
    / "srt"
    / "layers"
    / "quantization"
    / "activation_fallback.py"
)
if not _ACTIVATION_FALLBACK_PATH.is_file():
    warnings.warn(
        "tests/quantization requires "
        f"{_ACTIVATION_FALLBACK_PATH} to exist in the checkout; "
        "run the suite from a full repository checkout (no sglang install "
        "is required, but the source tree must be present). "
        "Install only tests/quantization/requirements.txt (pytest) and run "
        "pytest with tests/quantization as the rootdir; do not run "
        "`pip install -e python`.",
        RuntimeWarning,
        stacklevel=1,
    )

# The directory-local pytest.ini pins rootdir/testpaths to this directory so
# the suite does not depend on the repository-root pytest configuration.
_PYTEST_INI_PATH = Path(__file__).resolve().parent / "pytest.ini"
if not _PYTEST_INI_PATH.is_file():
    warnings.warn(
        "tests/quantization/pytest.ini is missing; run this suite with "
        "`-c tests/quantization/pytest.ini` so rootdir/testpaths are pinned "
        "to this directory and the repository-root configuration is not "
        "picked up.",
        RuntimeWarning,
        stacklevel=1,
    )


def pytest_configure(config: pytest.Config) -> None:
    """Warn (not fail) when the suite is misconfigured.

    The activation-fallback suite must run without importing ``sglang``. If a
    caller forgets ``-c tests/quantization/pytest.ini``, the repository-root
    pytest configuration may be picked up and pull in the full sglang runtime
    dependency set, which is exactly the environment-setup failure this suite
    is designed to avoid. Surface that as a clear warning so the environment
    is visible in CI logs, while still letting the self-contained tests run
    when the source tree is present.
    """

    if not _ACTIVATION_FALLBACK_PATH.is_file():
        warnings.warn(
            "tests/quantization requires "
            f"{_ACTIVATION_FALLBACK_PATH} to exist in the checkout; run the "
            "suite from a full repository checkout (no sglang install is "
            "required, but the source tree must be present). The "
            "activation-fallback tests will be skipped.",
            RuntimeWarning,
            stacklevel=1,
        )

    if not _PYTEST_INI_PATH.is_file():
        warnings.warn(
            "tests/quantization/pytest.ini is missing; run this suite with "
            "`-c tests/quantization/pytest.ini` so rootdir/testpaths are "
            "pinned to this directory and the repository-root configuration "
            "is not picked up.",
            RuntimeWarning,
            stacklevel=1,
        )

    # The suite is self-contained and must not depend on the full sglang
    # runtime. A full ``pip install -e python`` is not required, but it is
    # also not fatal: the tests load the module under test directly from its
    # file path and never import ``sglang`` themselves. Emit a warning so the
    # environment is visible in CI logs without blocking the run.
    if _sglang_installed_from_full_install():
        warnings.warn(
            "tests/quantization is running with a full sglang install "
            "present; the suite is self-contained and only requires "
            "tests/quantization/requirements.txt (pytest).",
            RuntimeWarning,
            stacklevel=1,
        )


def _sglang_installed_from_full_install() -> bool:
    """Return True when a full sglang install is present in the environment.

    The activation-fallback suite is self-contained and must not depend on the
    full sglang runtime. If ``sglang`` is importable, the environment was
    prepared with the full dependency set. We only check importability here;
    we never import ``sglang`` ourselves.
    """

    import importlib.util

    return importlib.util.find_spec("sglang") is not None
