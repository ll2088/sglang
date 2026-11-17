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

import sys
from pathlib import Path

# Make the repository root importable so tests can locate the source tree
# without relying on an installed ``sglang`` package.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
