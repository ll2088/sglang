#!/usr/bin/env bash
# Run the standalone DSA top-k BF16 capability tests with a minimal dependency
# set.  This script deliberately avoids `pip install -e .` / `pip install
# sglang`, which would resolve the full meta-package dependency tree.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" -m pip install --disable-pip-version-check -q \
  -r "${HERE}/requirements-minimal.txt"

cd "${HERE}"
"${PYTHON_BIN}" -m pytest -c "${HERE}/pytest.ini" "${HERE}/test_dsa_topk_bf16.py" "$@"
