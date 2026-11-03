#!/usr/bin/env bash
# Install the project without pulling the full runtime dependency tree, then
# install only the minimal test dependencies. Used by CI environment setup.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "error: python interpreter '${PYTHON_BIN}' not found" >&2
  exit 1
fi

echo "python: $(${PYTHON_BIN} --version 2>&1)"
echo "pip: $(${PYTHON_BIN} -m pip --version 2>&1)"

if [ -d "${REPO_ROOT}/python" ]; then
  PKG_DIR="${REPO_ROOT}/python"
else
  PKG_DIR="${REPO_ROOT}"
fi

set -x
"${PYTHON_BIN}" -m pip install --no-deps -e "${PKG_DIR}"
"${PYTHON_BIN}" -m pip install -r "${SCRIPT_DIR}/requirements-test-min.txt"
set +x

echo "test dependencies installed"
