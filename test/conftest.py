# Keep pytest collection for the stdlib-only Dockerfile permission checks
# isolated from the rest of the repository. This file intentionally imports
# nothing from sglang so `pytest test/test_dockerfile_permissions.py` can run
# in a minimal environment without installing the full sglang dependency tree.
#
# Do not add repository-wide fixtures or imports here; put them in a dedicated
# conftest.py closer to the tests that need them.

# Intentionally empty: the Dockerfile permission checks are stdlib-only and
# must not import sglang or mutate sys.path during collection. Mutating
# sys.path here previously risked pulling in the repository root package and
# triggering a full sglang dependency install in CI setup. Keep this file
# import-free so `pytest -c test/pytest.ini test/test_dockerfile_permissions.py`
# runs under a bare python3 interpreter.
