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

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
