#!/usr/bin/env bash
# Minimal environment preparation for the run_suite unit tests.
#
# Installs only test/unit/requirements.txt (pytest) and self-checks pytest so
# an environment problem is reported as an environment failure instead of
# being mistaken for a test failure. Must never trigger the full sglang
# dependency resolution (`pip install -e python`).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQUIREMENTS="${SCRIPT_DIR}/requirements.txt"

if [[ ! -f "${REQUIREMENTS}" ]]; then
  echo "[setup] ERROR: requirements file not found: ${REQUIREMENTS}" >&2
  exit 1
fi

echo "[setup] using $(realpath "${BASH_SOURCE[0]}")"
echo "[setup] requirements: $(realpath "${REQUIREMENTS}")"

# Guard: the minimal requirements file must stay pytest-only. If the full
# sglang dependency set sneaks back in, fail fast instead of resolving it.
if grep -Eq '^(sglang==|quack-kernels)' "${REQUIREMENTS}"; then
  echo "[setup] ERROR: ${REQUIREMENTS} must stay pytest-only" >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" -m pip install --disable-pip-version-check -r "${REQUIREMENTS}"

"${PYTHON_BIN}" -c 'import pytest; print("[setup] pytest", pytest.__version__)'
"${PYTHON_BIN}" -m pytest --version

echo "[setup] OK"
