#!/usr/bin/env python3
"""Standalone entry point for the stdlib-only Dockerfile permission checks.

This file is intentionally import-free of sglang and pytest at module scope so
CI can run it with a bare ``python3`` interpreter:

    python3 test/test_dockerfile_permissions.py

It also works under pytest:

    python3 -m pytest -q test/test_dockerfile_permissions.py

When pytest is not installed, running this file directly executes the checks
via ``checks.main()``.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CHECKS_DIR = os.path.join(_HERE, "dockerfile_permissions")
if _CHECKS_DIR not in sys.path:
    sys.path.insert(0, _CHECKS_DIR)

from checks import (  # noqa: E402  (re-exported for pytest collection)
    check_devcontainer_workspace_not_world_writable as test_devcontainer_workspace_not_world_writable,
    check_minimal_permission_statements as test_minimal_permission_statements,
    check_no_world_writable_permissions as test_no_world_writable_permissions,
    check_readme_keyword_hit_threshold as test_readme_keyword_hit_threshold,
    check_readme_non_root_cache_guidance as test_readme_non_root_cache_guidance,
)


if __name__ == "__main__":
    from checks import main as _main

    sys.exit(_main())
