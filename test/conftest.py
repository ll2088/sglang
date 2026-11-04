# Keep pytest collection for the stdlib-only Dockerfile permission checks
# isolated from the rest of the repository. This file intentionally imports
# nothing from sglang so `pytest test/test_dockerfile_permissions.py` can run
# in a minimal environment without installing the full sglang dependency tree.
#
# Do not add repository-wide fixtures or imports here; put them in a dedicated
# conftest.py closer to the tests that need them.

# The Dockerfile permission checks are stdlib-only and must not import
# sglang or mutate sys.path during collection. Mutating sys.path here
# previously risked pulling in the repository root package and triggering a
# full sglang dependency install in CI setup. Keep this file import-free so
# `pytest -c test/pytest.ini test/test_dockerfile_permissions.py` runs under a
# bare python3 interpreter.
#
# Explicitly drop the repository root from sys.path so a stray `import sglang`
# in a collected module cannot resolve to the in-tree package and start a
# dependency install. This is a no-op when the root is not on sys.path.
import sys as _sys

_REPO_ROOT = str(__file__.rsplit("/", 2)[0]) if "/" in __file__ else ""
if _REPO_ROOT and _REPO_ROOT in _sys.path:
    _sys.path.remove(_REPO_ROOT)
