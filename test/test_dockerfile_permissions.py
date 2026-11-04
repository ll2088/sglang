#!/usr/bin/env python3
"""Entry point for the stdlib-only Dockerfile permission checks.

Runs the checks directly when executed as a script, and exposes them as
``test_*`` functions for pytest. This module must not import sglang or any
third-party package so CI can run it with a bare ``python3`` interpreter.

Usage:
    python3 test/test_dockerfile_permissions.py
    python3 -m pytest -q test/test_dockerfile_permissions.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CHECKS_DIR = os.path.join(_HERE, "dockerfile_permissions")
if _CHECKS_DIR not in sys.path:
    sys.path.insert(0, _CHECKS_DIR)

from checks import (  # noqa: E402  (path set up above)
    ALL_CHECKS,
    check_devcontainer_workspace_not_world_writable,
    check_minimal_permission_statements,
    check_no_world_writable_permissions,
    check_readme_keyword_hit_threshold,
    check_readme_non_root_cache_guidance,
)


def test_no_world_writable_permissions():
    check_no_world_writable_permissions()


def test_minimal_permission_statements():
    check_minimal_permission_statements()


def test_devcontainer_workspace_not_world_writable():
    check_devcontainer_workspace_not_world_writable()


def test_readme_non_root_cache_guidance():
    check_readme_non_root_cache_guidance()


def test_readme_keyword_hit_threshold():
    check_readme_keyword_hit_threshold()


def test_all_checks_pass():
    for check in ALL_CHECKS:
        check()


def main():
    failures = 0
    for check in ALL_CHECKS:
        try:
            check()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {check.__name__}: {exc}", file=sys.stderr)
        else:
            print(f"PASS {check.__name__}")
    if failures:
        print(f"{failures} check(s) failed", file=sys.stderr)
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
