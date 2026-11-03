#!/usr/bin/env python3
"""Static pattern checks for the SonarQube-inspired fixes.

This script is intentionally dependency-free so it can run before the test
environment is fully provisioned. It reports (and fails on) the patterns the
upstream PR removed:

* mutable default arguments (dict/list/set literals in signatures)
* ``hasattr`` checks on dataclass fields
* ``weights_only=False`` pickle fallbacks

It is a best-effort textual check, not a full AST analysis.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Files that are expected to contain the fixed patterns (tests intentionally
# exercise the old behaviour).
EXCLUDE_DIRS = {".git", "build", "dist", "node_modules", "__pycache__"}

MUTABLE_DEFAULT_RE = re.compile(
    r"def\s+\w+\s*\([^)]*=\s*(?:\{\}|\[\]|set\(\))", re.DOTALL
)
HASATTR_DATACLASS_RE = re.compile(r"hasattr\(\s*\w+\s*,\s*['\"]\w+['\"]\s*\)")
WEIGHTS_ONLY_FALSE_RE = re.compile(r"weights_only\s*=\s*False")


def iter_python_files(root: Path):
    for path in root.rglob("*.py"):
        if any(part in EXCLUDE_DIRS for part in path.parts):
            continue
        yield path


def main() -> int:
    failures: list[str] = []
    for path in iter_python_files(REPO_ROOT):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        rel = path.relative_to(REPO_ROOT)
        for lineno, line in enumerate(text.splitlines(), start=1):
            if MUTABLE_DEFAULT_RE.search(line):
                failures.append(f"{rel}:{lineno}: mutable default argument")
            if WEIGHTS_ONLY_FALSE_RE.search(line):
                failures.append(f"{rel}:{lineno}: weights_only=False")
            if HASATTR_DATACLASS_RE.search(line) and "dataclass" in text:
                failures.append(f"{rel}:{lineno}: hasattr on dataclass field")

    if failures:
        print("Static pattern check failed:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1

    print("Static pattern check passed: no mutable defaults, "
          "weights_only=False, or dataclass hasattr patterns found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
