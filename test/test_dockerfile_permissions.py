#!/usr/bin/env python3
"""Standalone + pytest entry point for the stdlib-only Dockerfile checks.

This file is intentionally import-free of sglang and pytest at module scope so
CI can run it with a bare ``python3`` interpreter:

    python3 test/test_dockerfile_permissions.py
    python3 -m pytest -q test/test_dockerfile_permissions.py

The actual assertions live in ``test/dockerfile_permissions/checks.py`` so the
same logic is shared with ``test/dockerfile_permissions/run.py`` and the pytest
wrapper in ``test/dockerfile_permissions/test_dockerfile_permissions.py``.
"""

import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CHECKS_PATH = os.path.join(_HERE, "dockerfile_permissions", "checks.py")

if not os.path.isfile(_CHECKS_PATH):
    raise ImportError(
        "test/dockerfile_permissions/checks.py is missing; the Dockerfile "
        "permission checks must remain stdlib-only"
    )

_spec = importlib.util.spec_from_file_location("dockerfile_permissions_checks", _CHECKS_PATH)
_checks = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_checks)

# Re-export the checks as test_* functions for pytest collection.
test_no_world_writable_permissions = _checks.check_no_world_writable_permissions
test_minimal_permission_statements = _checks.check_minimal_permission_statements
test_devcontainer_workspace_not_world_writable = _checks.check_devcontainer_workspace_not_world_writable
test_readme_non_root_cache_guidance = _checks.check_readme_non_root_cache_guidance
test_readme_keyword_hit_threshold = _checks.check_readme_keyword_hit_threshold


def test_all_checks_pass():
    """Run every check so a single pytest invocation covers the full suite."""
    _checks.run_all()


if __name__ == "__main__":
    _checks.run_all()
    print("dockerfile permission checks passed")
