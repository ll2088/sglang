#!/usr/bin/env python3
"""Standalone entry point for the Dockerfile permission checks.

This runner is stdlib-only and must never import sglang or pytest. It is the
preferred CI entry point because it does not depend on pytest collection and
cannot accidentally pick up the repository-root pytest configuration.

Usage:
    cd test/dockerfile_permissions && python3 run.py
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Keep the repository root off sys.path so a stray `import sglang` cannot
# resolve to the in-tree package and start a dependency install.
_REPO_ROOT = str(_HERE.parent.parent)
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
if "sglang" not in sys.modules:
    sys.modules["sglang"] = None

from checks import CHECKS, missing_paths  # noqa: E402


def _readme_keyword_hit_count() -> int:
    """Count README keyword hits the same way the acceptance grep does."""
    from checks import READMES

    keywords = (
        "cache",
        "缓存",
        "chown",
        "chmod",
        "go-w",
        "非默认用户",
        "non-root",
    )
    total = 0
    for readme in READMES:
        content = readme.read_text(encoding="utf-8")
        total += sum(1 for keyword in keywords if keyword in content)
    return total


def main() -> int:
    missing = missing_paths()
    if missing:
        for path in missing:
            print(f"FAIL missing expected file: {path}", file=sys.stderr)
        return 1

    failures = 0
    for check in CHECKS:
        try:
            check()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {check.__name__}: {exc}", file=sys.stderr)
        else:
            print(f"PASS {check.__name__}")

    hits = _readme_keyword_hit_count()
    if hits < 6:
        failures += 1
        print(
            f"FAIL readme_keyword_hits: expected >= 6, found {hits}",
            file=sys.stderr,
        )
    else:
        print(f"PASS readme_keyword_hits ({hits})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
