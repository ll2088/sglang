#!/usr/bin/env bash
# Minimal environment preparation for the run_suite verification dimensions.
#
# The verification jobs only exercise run_suite.py's exit-code and
# collection-diagnostic branches; conftest.py stubs out sglang.test.ci.* so the
# full sglang runtime (torch, kernels, ...) is not needed. Installing the full
# dependency set with `pip install -e python` resolves torch/kernels and can
# fail before any test runs, which is what previously broke every dimension at
# the environment-preparation step.
#
# Usage (from the repository root):
#
#     bash test/unit/setup_env.sh
#     python test/unit/verify_run_suite.py
set -euo pipefail

# Print the resolved script path so CI logs prove this lightweight entry was
# actually hit instead of a full `pip install -e python`.
echo "[setup] using $(realpath "$0")"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REQUIREMENTS="${SCRIPT_DIR}/requirements.txt"

if [ ! -f "${REQUIREMENTS}" ]; then
    echo "[setup] missing ${REQUIREMENTS}" >&2
    exit 1
fi

# Guard: the minimal requirements file must stay pytest-only. If the full
# sglang dependency set comes back here, fail loudly instead of silently
# resolving torch/kernels during environment preparation.
if grep -Eq '^(sglang==|quack-kernels)' "${REQUIREMENTS}"; then
    echo "[setup] ${REQUIREMENTS} must not pin the full sglang dependency set" >&2
    exit 1
fi

PYTHON_BIN="${PYTHON:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    PYTHON_BIN="python"
fi

echo "[setup] installing minimal requirements from ${REQUIREMENTS}"
"${PYTHON_BIN}" -m pip install -q -r "${REQUIREMENTS}"

# Self-check so an environment problem is reported as an environment failure
# instead of being mistaken for a test failure.
echo "[setup] self-check: import pytest"
"${PYTHON_BIN}" -c 'import pytest'

echo "[setup] self-check: pytest --version"
"${PYTHON_BIN}" -m pytest --version

echo "[setup] ready (repo root: ${REPO_ROOT})"
