"""Local conftest for the standalone DSA top-k BF16 capability tests.

These tests are deliberately dependency-light: they only need the standard
library and the pure-Python helper module under
``python/sglang/kernels/dsa_topk_bf16_capability.py``.  The module is loaded by
an explicit file path in the test itself, so this conftest only keeps the
repository root importable and never imports the ``sglang`` package (which
would resolve its full dependency set).
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
