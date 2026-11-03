#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Install the minimal dependency set needed by the quant regression tests.
# This intentionally avoids `pip install -e python`, which pulls the full
# sglang dependency tree and fails in constrained CI environments.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONSTRAINTS="${REPO_ROOT}/constraints-ci.txt"
REQUIREMENTS="${REPO_ROOT}/requirements-test-min.txt"

if [[ ! -f "${CONSTRAINTS}" ]]; then
  echo "error: missing constraints file: ${CONSTRAINTS}" >&2
  exit 1
fi
if [[ ! -f "${REQUIREMENTS}" ]]; then
  echo "error: missing requirements file: ${REQUIREMENTS}" >&2
  exit 1
fi

echo "Using constraints file: ${CONSTRAINTS}"
echo "Using requirements file: ${REQUIREMENTS}"

python -m pip install --upgrade pip
python -m pip install \
  --constraint "${CONSTRAINTS}" \
  --requirement "${REQUIREMENTS}"

echo "Minimal test dependencies installed successfully."
