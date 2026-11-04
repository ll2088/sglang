#!/usr/bin/env python3
"""Standalone runner for the Dockerfile permission checks.

Stdlib-only: never imports pytest or sglang, so CI can run it with a bare
python3 interpreter without installing the sglang dependency tree.
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from checks import (  # noqa: E402
    CHECKS,
    missing_paths,
    readme_keyword_hit_count,
    readme_keyword_hits_per_file,
)


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

    hits = readme_keyword_hit_count()
    if hits < 6:
        failures += 1
        print(
            f"FAIL readme_keyword_hits: expected >= 6, found {hits}",
            file=sys.stderr,
        )
    else:
        print(f"PASS readme_keyword_hits ({hits})")
    for name, count in readme_keyword_hits_per_file().items():
        print(f"INFO readme_keyword_hits[{name}]={count}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
