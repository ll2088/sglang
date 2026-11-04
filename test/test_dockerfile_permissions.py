#!/usr/bin/env python3
"""Compatibility entry point for the Dockerfile permission checks.

This file exists so CI can invoke the checks with the historical path
``test/test_dockerfile_permissions.py`` without installing sglang. It is
stdlib-only and delegates to ``test/dockerfile_permissions/checks.py``.

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

from checks import (  # noqa: E402
    check_devcontainer_workspace_not_world_writable as test_devcontainer_workspace_not_world_writable,
    check_minimal_permission_statements as test_minimal_permission_statements,
    check_no_world_writable_permissions as test_no_world_writable_permissions,
    check_readme_keyword_hit_threshold as test_readme_keyword_hit_threshold,
    check_readme_non_root_cache_guidance as test_readme_non_root_cache_guidance,
    main as _main,
)


if __name__ == "__main__":
    sys.exit(_main())
