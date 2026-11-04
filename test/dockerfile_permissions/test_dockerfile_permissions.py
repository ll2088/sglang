"""Pytest entry point for the stdlib-only Dockerfile permission checks.

Run with an explicit config so the repository root is never consulted:

    python3 -m pytest -q -c test/dockerfile_permissions/pytest.ini \
        --rootdir test/dockerfile_permissions \
        test/dockerfile_permissions/test_dockerfile_permissions.py

or use the standalone runner: python3 test/dockerfile_permissions/run.py
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from checks import (  # noqa: E402,F401
    CHECKS,
    DOCKERFILES,
    HARDENED_DOCKERFILES,
    READMES,
    REPO_ROOT,
    missing_paths,
    readme_keyword_hit_count,
    readme_keyword_hits_per_file,
    test_devcontainer_workspace_not_world_writable,
    test_minimal_permission_statements,
    test_no_world_writable_permissions,
    test_readme_non_root_cache_guidance,
)
