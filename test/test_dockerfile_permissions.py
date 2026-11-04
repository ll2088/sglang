import sys
from pathlib import Path

# This module is a thin compatibility shim. The canonical, stdlib-only
# implementation lives in test/dockerfile_permissions/ so CI can run it with
# `python3 test/dockerfile_permissions/run.py` or
# `-c test/dockerfile_permissions/pytest.ini --rootdir
# test/dockerfile_permissions`, which never walks up to the repository root
# conftest.py or pyproject.toml and therefore never triggers a sglang install.
#
# Keep this file free of any `import sglang` / `import pytest` at module scope.
# It only re-exports the checks from the isolated package so callers that still
# invoke `python3 test/test_dockerfile_permissions.py` keep working.

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "dockerfile_permissions"))

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

# Backwards-compatible aliases for callers that used the old private names.
_readme_keyword_hit_count = readme_keyword_hit_count
_readme_keyword_hits_per_file = readme_keyword_hits_per_file
_missing_paths = missing_paths


def main() -> int:
    from run import main as _main

    return _main()


if __name__ == "__main__":
    sys.exit(main())
