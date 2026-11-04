# Keep pytest collection for the stdlib-only Dockerfile permission checks
# isolated from the rest of the repository. This file intentionally imports
# nothing from sglang so `pytest test/test_dockerfile_permissions.py` can run
# in a minimal environment without installing the full sglang dependency tree.
#
# The canonical checks live in test/dockerfile_permissions/ with their own
# pytest.ini and run.py. CI must prefer
# `cd test/dockerfile_permissions && python3 run.py` or
# `python3 -m pytest -q -c test/dockerfile_permissions/pytest.ini
# --rootdir test/dockerfile_permissions
# test/dockerfile_permissions/test_dockerfile_permissions.py` so this conftest
# is never loaded at all.
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
# CI must invoke the checks with `-c test/dockerfile_permissions/pytest.ini`
# (or run the standalone runner) so the repository-root pytest configuration is
# not picked up. The explicit `rootdir = .` in that pytest.ini pins the rootdir
# to the isolated directory so pytest does not walk up to the repository root
# and load a root-level conftest.py or pyproject.toml that could trigger a full
# sglang install.
#
# If a future CI job still runs `pip install -e python` before pytest, that is
# a setup-step problem outside this file: the permission/documentation checks
# need only a bare python3 (or python3 + pytest) and must not be gated on the
# full sglang dependency tree.
#
# This file must remain import-free. Do not add `import sglang`, `import pytest`,
# or any third-party import here: pytest loads conftest.py before collecting the
# test module, so a heavy import would reintroduce the full dependency install
# that the stdlib-only checks are designed to avoid.
