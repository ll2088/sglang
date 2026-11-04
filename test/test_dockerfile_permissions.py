#!/usr/bin/env python3
"""Stdlib-only entrypoint for the Dockerfile permission/documentation checks.

This module exists so CI can run the checks without installing sglang or any
third-party dependency:

    python3 test/test_dockerfile_permissions.py
    python3 -m pytest -q test/test_dockerfile_permissions.py

The actual check logic lives in ``test/dockerfile_permissions/checks.py`` and
only uses the Python standard library. This file deliberately avoids importing
pytest or sglang at module scope so it works under a bare ``python3``
interpreter; when pytest is available the checks are also exposed as
``test_*`` functions for collection.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CHECKS_DIR = os.path.join(_HERE, "dockerfile_permissions")
if _CHECKS_DIR not in sys.path:
    sys.path.insert(0, _CHECKS_DIR)

from checks import (  # noqa: E402  (path set up above)
    check_devcontainer_workspace_not_world_writable as test_devcontainer_workspace_not_world_writable,
    check_minimal_permission_statements as test_minimal_permission_statements,
    check_no_world_writable_permissions as test_no_world_writable_permissions,
    check_readme_keyword_hit_threshold as test_readme_keyword_hit_threshold,
    check_readme_non_root_cache_guidance as test_readme_non_root_cache_guidance,
)


def main():
    """Run every check without pytest and return a process exit code."""
    from checks import main as _main

    return _main()


if __name__ == "__main__":
    sys.exit(main())
