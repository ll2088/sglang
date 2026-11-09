#!/usr/bin/env bash
# Lightweight environment preparation for the run_suite verification jobs.
#
# The four verification dimensions (collect-diagnostics, exit-codes, tests,
# usability) only exercise run_suite.py's exit-code and collection-diagnostic
# branches. conftest.py stubs out sglang.test.ci.*, so the full sglang runtime
# (torch, kernels, ...) is NOT needed. Installing the package with
# `pip install -e python` resolves the full dependency set and can fail during
# environment preparation (e.g. torch wheel download timeouts), which is what
# previously made every dimension fail before any test ran.
#
# Usage (from the repository root):
#
#     bash test/unit/setup_env.sh
#     PYTHONPATH=test python -m pytest test/unit/test_run_suite_exit_codes.py -q -p no:cacheprovider
#
# This script is intentionally dependency-free beyond pip/pytest.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"

# 1) Install only the minimal dependency set (pytest). Never install the full
#    sglang dependency set here: it is not required by these tests and its
#    resolution is the failure mode this script exists to avoid.
"$PYTHON" -m pip install --disable-pip-version-check -r test/unit/requirements.txt

# 2) Fast self-check so a dependency problem is reported as an environment
#    failure instead of being mistaken for a test failure.
"$PYTHON" -c "import pytest; print('pytest', pytest.__version__)"
"$PYTHON" -m pytest --version

echo "run_suite verification environment ready (pytest only)"
