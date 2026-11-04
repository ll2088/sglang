#!/usr/bin/env python3
"""Standalone runner for the stdlib-only Dockerfile permission checks.

This entry point intentionally imports nothing outside the Python standard
library so CI can run it with a bare ``python3`` interpreter, without
installing sglang or pytest. It mirrors the pytest wrapper in
``test_dockerfile_permissions.py`` but reports failures directly.

Usage:
    python3 test/dockerfile_permissions/run.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from checks import ALL_CHECKS  # noqa: E402  (path set up above)


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
