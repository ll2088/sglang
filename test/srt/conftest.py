"""Test bootstrap for the lightweight SMG/gRPC perf unit tests.

These tests only need the standard library plus pytest. When the full
``sglang`` runtime is not installed, ``_bootstrap_light`` installs stub
packages so the target modules can be imported from source directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import _bootstrap_light  # noqa: E402

_bootstrap_light.install()
