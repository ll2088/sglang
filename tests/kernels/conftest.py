"""Local conftest for the standalone DSA top-k BF16 capability tests.

These tests are deliberately dependency-light: they only need the standard
library and the pure-Python helper module under
``python/sglang/kernels/jit/csrc/dsa/``.  This conftest makes the repository
root importable without installing the ``sglang`` package (and therefore
without resolving its full dependency set), and it keeps the tests runnable
when optional plugins such as ``pytest-randomly`` are not installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# The pure-Python capability module lives under the kernels package tree but
# must be importable without importing the sglang package itself (which would
# pull in torch/CUDA and the full dependency set).  Add its directory directly
# so `import dsa_topk_bf16_capability` resolves to the standalone module.
_KERNELS_DIR = _REPO_ROOT / "python" / "sglang" / "kernels"
if str(_KERNELS_DIR) not in sys.path:
    sys.path.insert(0, str(_KERNELS_DIR))
