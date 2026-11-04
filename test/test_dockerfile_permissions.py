#!/usr/bin/env python3
"""Thin shim so the Dockerfile permission checks can be run from the
repository root without importing sglang or any third-party package.

This file exists because CI invokes:
    python3 test/test_dockerfile_permissions.py
    python3 -m pytest -q test/test_dockerfile_permissions.py

Both forms must work with a bare python3 interpreter. The real check logic
lives in ``test/dockerfile_permissions/checks.py`` and is stdlib-only.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CHECKS_DIR = os.path.join(_HERE, "dockerfile_permissions")
if _CHECKS_DIR not in sys.path:
    sys.path.insert(0, _CHECKS_DIR)

from checks import ALL_CHECKS  # noqa: E402  (path set up above)


def test_all_checks_pass():
    """Run every check so a single pytest invocation covers the full suite."""
    for check in ALL_CHECKS:
        check()


def main() -> int:
    failures = []
    for check in ALL_CHECKS:
        name = getattr(check, "__name__", repr(check))
        try:
            check()
        except AssertionError as exc:
            failures.append((name, str(exc)))
        except Exception as exc:  # noqa: BLE001 - report any check error
            failures.append((name, f"{type(exc).__name__}: {exc}"))

    if failures:
        for name, message in failures:
            print(f"FAIL {name}: {message}", file=sys.stderr)
        return 1

    print(f"OK: {len(ALL_CHECKS)} Dockerfile permission checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
