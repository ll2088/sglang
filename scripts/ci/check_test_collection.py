#!/usr/bin/env python3
"""Fail fast if the quant regression tests cannot be collected.

This script is intentionally dependency-light (stdlib + pytest) so it can run
before the full sglang runtime is installed.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

TARGET_TESTS = [
    REPO_ROOT / "test/registered/quant/test_quant_utils_defaults.py",
    REPO_ROOT / "test/registered/quant/test_dataclass_optional_fields.py",
    REPO_ROOT / "test/registered/quant/test_pickle_loading_safety.py",
    REPO_ROOT / "test/registered/quant/test_moe_init_keywords.py",
]


def main() -> int:
    missing = [str(path) for path in TARGET_TESTS if not path.is_file()]
    if missing:
        print("ERROR: missing test files:", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        return 1

    cmd = [sys.executable, "-m", "pytest", "--collect-only", "-q"]
    cmd.extend(str(path) for path in TARGET_TESTS)
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        print(
            "ERROR: pytest collection failed; see output above",
            file=sys.stderr,
        )
        return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
