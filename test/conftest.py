# Keep pytest collection for the stdlib-only Dockerfile permission checks
# isolated from the rest of the repository. This file intentionally imports
# nothing from sglang so `pytest test/test_dockerfile_permissions.py` can run
# in a minimal environment without installing the full sglang dependency tree.
#
# Do not add repository-wide fixtures or imports here; put them in a dedicated
# conftest.py closer to the tests that need them.
#
# The checks in test_dockerfile_permissions.py are stdlib-only and never import
# sglang. This conftest deliberately does not touch sys.path or sys.modules:
# mutating them during collection is fragile (it depends on how pytest was
# invoked and on the caller's CWD) and cannot prevent a CI setup step from
# running `pip install -e python` before pytest starts. Keeping this file
# import-free means collection works under a bare python3 interpreter and
# cannot itself trigger a dependency install.
#
# CI must invoke the checks with `-c test/pytest.ini` (or run the standalone
# runner) so the repository-root pytest configuration is not picked up.
