#!/usr/bin/env bash
# Install only the minimal dependencies needed by the self-contained quant
# regression tests. Deliberately avoids `pip install -e python`, which pulls
# the full sglang dependency set (torch, triton, nvidia-*, ...).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

python -m pip install --upgrade pip
python -m pip install \
  -r "${SCRIPT_DIR}/requirements-test-min.txt" \
  -c "${SCRIPT_DIR}/constraints-ci.txt"

echo "Minimal test dependencies installed."
echo "Run tests with: PYTHONPATH=${REPO_ROOT}/python python -m pytest -q test/registered/quant"
