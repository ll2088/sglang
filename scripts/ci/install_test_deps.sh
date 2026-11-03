#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Install the minimal dependency set needed by the quantization regression
# tests without resolving the full sglang dependency tree.
set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" --version
"${PYTHON_BIN}" -m pip --version

# Install the minimal test requirements with the shared constraints file.
# --no-deps prevents pip from pulling in the full sglang dependency graph.
"${PYTHON_BIN}" -m pip install \
  --no-deps \
  -c "${SCRIPT_DIR}/constraints-ci.txt" \
  -r "${SCRIPT_DIR}/requirements-test-min.txt"

# Install the project in editable mode without dependencies so that the
# sglang package is importable from the repository checkout.
"${PYTHON_BIN}" -m pip install \
  --no-deps \
  --no-build-isolation \
  -e "${REPO_ROOT}/python"

echo "Minimal test dependencies installed successfully."
