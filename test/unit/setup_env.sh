#!/usr/bin/env bash
# Minimal environment setup for the run_suite unit tests.
#
# These tests only exercise run_suite.py's exit-code and collection-diagnostic
# branches; test/unit/conftest.py stubs out sglang.test.ci.* so the full sglang
# runtime (torch, kernels, ...) is not needed. CI must call this script instead
# of `pip install -e python`, which resolves the full dependency set and can
# fail during environment preparation before any test runs.
#
# Usage (from the repository root):
#
#     bash test/unit/setup_env.sh
#     PYTHONPATH=test python test/unit/verify_run_suite.py
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REQ_FILE="${SCRIPT_DIR}/requirements.txt"

if [[ ! -f "${REQ_FILE}" ]]; then
    echo "setup_env: missing requirements file: ${REQ_FILE}" >&2
    exit 2
fi

PYTHON_BIN="${PYTHON:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "setup_env: python interpreter not found: ${PYTHON_BIN}" >&2
    exit 2
fi

echo "setup_env: installing minimal test dependencies from ${REQ_FILE}"
if ! "${PYTHON_BIN}" -m pip install -r "${REQ_FILE}"; then
    echo "setup_env: pip install failed for ${REQ_FILE}" >&2
    exit 2
fi

# Self-check: an environment problem must be reported as an environment
# failure (exit 2), never mistaken for a test failure.
if ! "${PYTHON_BIN}" -c 'import pytest'; then
    echo "setup_env: pytest is not importable after install" >&2
    exit 2
fi
if ! "${PYTHON_BIN}" -m pytest --version; then
    echo "setup_env: pytest --version failed" >&2
    exit 2
fi

# Guard against regressions that would pull the full sglang dependency set
# back into environment preparation.
if grep -Eq 'sglang==|quack-kernels|^torch' "${REQ_FILE}"; then
    echo "setup_env: ${REQ_FILE} must stay pytest-only (found full sglang deps)" >&2
    exit 2
fi

echo "setup_env: OK (pytest-only environment ready)"
