# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal conftest for the activation-fallback test suite.

This suite is intentionally self-contained: it must run without installing the
``sglang`` package (no GPU, no compiled kernels, no third-party deps). The
conftest therefore only ensures the repository root is importable and does not
import anything from ``sglang``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# This suite is intentionally self-contained: it must run without installing
# the ``sglang`` package (no GPU, no compiled kernels, no third-party deps).
# The conftest therefore only ensures the repository root is importable and
# does not import anything from ``sglang``.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Also expose the ``python`` source tree so that, if a future test in this
# directory does need to import ``sglang``, it can do so from the checkout
# without a prior ``pip install -e python``. Importing ``sglang`` itself is
# still not required by the activation-fallback suite.
_PYTHON_ROOT = _REPO_ROOT / "python"
if _PYTHON_ROOT.is_dir() and str(_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYTHON_ROOT))

# Fail fast with an actionable diagnostic if the module under test is missing.
# CI logs otherwise show an opaque FileNotFoundError from the test module.
#
# This suite is intentionally self-contained: it must run without installing
# the ``sglang`` package (no GPU, no compiled kernels, no third-party deps).
# CI jobs that run this directory should install only
# ``tests/quantization/requirements.txt`` (pytest) and must NOT run
# ``pip install -e python``, which pulls the full sglang runtime dependency
# set and fails on GPU-less runners.
#
# The directory-local pytest.ini pins rootdir/testpaths to this directory so
# the suite does not depend on the repository-root pytest configuration.
_PYTEST_INI_PATH = Path(__file__).resolve().parent / "pytest.ini"
if not _PYTEST_INI_PATH.is_file():
    import warnings

    warnings.warn(
        "tests/quantization/pytest.ini is missing; run this suite with "
        "`-c tests/quantization/pytest.ini` so rootdir/testpaths are pinned "
        "to this directory and the repository-root configuration is not "
        "picked up.",
        RuntimeWarning,
        stacklevel=1,
    )

_ACTIVATION_FALLBACK_PATH = (
    _PYTHON_ROOT
    / "sglang"
    / "srt"
    / "layers"
    / "quantization"
    / "activation_fallback.py"
)
if not _ACTIVATION_FALLBACK_PATH.is_file():
    # Emit a clear, actionable diagnostic instead of crashing the whole
    # pytest session with an opaque traceback. The individual test module
    # also checks for the file and will fail with a precise message.
    import warnings

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
