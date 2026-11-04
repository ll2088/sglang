# Stdlib-only Dockerfile permission/documentation checks.
#
# This package is intentionally import-free of sglang and pytest at module
# scope so CI can run it with a bare python3 interpreter. The canonical entry
# points are:
#   python3 test/dockerfile_permissions/run.py
#   python3 -m pytest -q -c test/dockerfile_permissions/pytest.ini \
#       test/dockerfile_permissions/
#
# Keeping the checks in their own directory with their own pytest.ini pins the
# rootdir here, so pytest never walks up to the repository root and loads a
# root-level conftest.py or pyproject.toml that could trigger a full sglang
# dependency install during collection.
