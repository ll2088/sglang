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

# Guard against accidental full-package imports: if a future test in this
# directory tries to import sglang, surface a clear message instead of a
# confusing ImportError from a partially installed editable package.
import builtins as _builtins  # noqa: E402

_real_import = _builtins.__import__


def _guarded_import(name, *args, **kwargs):
    if name == "sglang" or name.startswith("sglang."):
        raise ImportError(
            "tests/quantization is a self-contained suite and must not import "
            "the sglang package. Load the module under test directly from its "
            "file path instead (see test_activation_fallback.py)."
        )
    return _real_import(name, *args, **kwargs)


_builtins.__import__ = _guarded_import
