#!/usr/bin/env bash
# Install the minimal dependency set required by the registered quant tests.
# Uses --no-deps plus a constraints file so pip never resolves the full
# sglang dependency tree (py-spy, quack-kernels, ...), which previously made
# CI environment setup fail before pytest could run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "error: python interpreter '${PYTHON_BIN}' not found" >&2
  exit 1
fi

CONSTRAINTS="${SCRIPT_DIR}/constraints-ci.txt"
REQUIREMENTS="${SCRIPT_DIR}/requirements-test-min.txt"

for f in "${CONSTRAINTS}" "${REQUIREMENTS}"; do
  if [[ ! -f "${f}" ]]; then
    echo "error: required file not found: ${f}" >&2
    exit 1
  fi
done

set -x
"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install --no-deps -c "${CONSTRAINTS}" -r "${REQUIREMENTS}"
# Install the project itself without resolving dependencies.
if [[ -f "${REPO_ROOT}/pyproject.toml" ]]; then
  "${PYTHON_BIN}" -m pip install --no-deps -e "${REPO_ROOT}"
fi
"${PYTHON_BIN}" -m pip freeze
set +x

echo "minimal test dependencies installed"
