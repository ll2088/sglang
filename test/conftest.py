# Keep pytest collection for the stdlib-only Dockerfile permission checks
# isolated from the rest of the repository. This file intentionally imports
# nothing from sglang so `pytest test/test_dockerfile_permissions.py` can run
# in a minimal environment without installing the full sglang dependency tree.
#
# Do not add repository-wide fixtures or imports here; put them in a dedicated
# conftest.py closer to the tests that need them.
