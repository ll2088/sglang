#!/usr/bin/env bash
# Minimal environment setup for the run_suite unit tests.
#
# These tests only exercise run_suite.py's exit-code and collection-diagnostic
# branches; test/unit/conftest.py stubs out sglang.test.ci.* so the full
# sglang runtime (torch, kernels, ...) is not needed. CI must install only
# test/unit/requirements.txt instead of `pip install -e python`, which
# resolves the full dependency set and can fail before any test runs.
#
# Usage (from the repository root):
#
#     bash test/unit/setup_env.sh
#     python test/unit/verify_run_suite.py
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REQ_FILE="${SCRIPT_DIR}/requirements.txt"

# Print the resolved path so CI logs prove this lightweight entrypoint was
# actually hit instead of silently falling back to the full dependency set.
echo "[setup_env] script=${SCRIPT_DIR}/setup_env.sh"
echo "[setup_env] repo_root=${REPO_ROOT}"
echo "[setup_env] requirements=${REQ_FILE}"

if [[ ! -f "${REQ_FILE}" ]]; then
    echo "[setup_env] ERROR: missing ${REQ_FILE}" >&2
    exit 2
fi

# Guard: the minimal requirements file must stay pytest-only. If the full
# sglang dependency set creeps back in, fail loudly instead of resolving it.
if grep -Eq '^[[:space:]]*(sglang==|quack-kernels|torch)' "${REQ_FILE}"; then
    echo "[setup_env] ERROR: ${REQ_FILE} must not pin the full sglang deps" >&2
    exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    PYTHON_BIN="python"
fi

echo "[setup_env] installing minimal deps with ${PYTHON_BIN} -m pip"
"${PYTHON_BIN}" -m pip install --disable-pip-version-check -r "${REQ_FILE}"

# Self-check: an environment problem must be reported as an environment
# failure (exit 2) rather than being mistaken for a test failure.
echo "[setup_env] self-check: import pytest"
if ! "${PYTHON_BIN}" -c 'import pytest'; then
    echo "[setup_env] ERROR: pytest is not importable after install" >&2
    exit 2
fi

echo "[setup_env] self-check: pytest --version"
if ! "${PYTHON_BIN}" -m pytest --version; then
    echo "[setup_env] ERROR: pytest --version failed" >&2
    exit 2
fi

echo "[setup_env] OK: pytest-only environment ready"
