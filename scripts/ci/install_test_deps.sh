#!/usr/bin/env bash
# Install the minimal dependency set required by the regression tests.
# This intentionally avoids `pip install -e python`, which resolves the full
# sglang runtime dependency tree and is not needed for these self-contained
# tests.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REQ_FILE="${SCRIPT_DIR}/requirements-test-min.txt"

if [[ ! -f "${REQ_FILE}" ]]; then
  echo "ERROR: missing minimal requirements file: ${REQ_FILE}" >&2
  exit 1
fi

echo "Installing minimal test dependencies from ${REQ_FILE}"
python -m pip install --no-deps -r "${REQ_FILE}"

echo "Minimal test dependencies installed."
