#!/usr/bin/env bash
# Install only the minimal dependencies needed by the quant regression tests.
# Avoids `pip install -e .`, which resolves the full sglang dependency set.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install \
  -r "${SCRIPT_DIR}/requirements-test-min.txt" \
  -c "${SCRIPT_DIR}/constraints-ci.txt"

# Install the local package without pulling its runtime dependencies.
"${PYTHON_BIN}" -m pip install --no-deps -e "${REPO_ROOT}/python"

# Self-check so a broken environment fails fast with a clear message.
"${PYTHON_BIN}" -c "import pytest; print('pytest', pytest.__version__)"
"${PYTHON_BIN}" -c "import torch; print('torch', torch.__version__)"
