#!/usr/bin/env bash
# Lightweight CI dependency installer.
# Installs only the minimal test requirements without resolving the sglang
# package metadata (which would pull heavy runtime deps such as py-spy,
# pyzmq and quack-kernels).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ_FILE="${SCRIPT_DIR}/requirements-test-min.txt"
CONSTRAINTS_FILE="${SCRIPT_DIR}/constraints-ci.txt"

if [[ ! -f "${REQ_FILE}" ]]; then
  echo "ERROR: missing requirements file: ${REQ_FILE}" >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
echo "Using ${PYTHON_BIN} ($(${PYTHON_BIN} --version 2>&1))"

# --no-deps prevents pip from resolving transitive dependencies of any
# package that might reference sglang; the minimal list is self-contained.
if ! "${PYTHON_BIN}" -m pip install --no-deps \
    -r "${REQ_FILE}" \
    -c "${CONSTRAINTS_FILE}"; then
  echo "ERROR: failed to install lightweight test dependencies" >&2
  echo "Requirements: ${REQ_FILE}" >&2
  echo "Constraints:  ${CONSTRAINTS_FILE}" >&2
  exit 1
fi

echo "Lightweight test dependencies installed successfully."
