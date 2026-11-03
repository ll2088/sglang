#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Install test dependencies for the quantization regression suite.
#
# The CI environment must not resolve the full sglang dependency tree; we
# install the package with --no-deps and pin the handful of runtime deps the
# tests actually import via constraints-ci.txt.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONSTRAINTS="${REPO_ROOT}/constraints-ci.txt"

if [[ ! -f "${CONSTRAINTS}" ]]; then
  echo "constraints file not found: ${CONSTRAINTS}" >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python}"

PIP_CMD=(
  "${PYTHON_BIN}" -m pip install
  --no-deps
  -c "${CONSTRAINTS}"
  -e "${REPO_ROOT}/python"
)

echo "constraints file: ${CONSTRAINTS}"
echo "pip command: ${PIP_CMD[*]}"

"${PIP_CMD[@]}"

# Smoke check: the package must import without pulling in optional deps.
echo "smoke check: import sglang"
"${PYTHON_BIN}" -c "import sglang; print('sglang import ok')"

echo "test deps installed"
