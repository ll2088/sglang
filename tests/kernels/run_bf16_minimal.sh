#!/usr/bin/env bash
# Run the standalone DSA top-k BF16 capability tests without installing the
# sglang meta-package (which would resolve py-spy/pybase64/pydantic/...).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
REQ="${HERE}/requirements-minimal.txt"

if [[ ! -f "${REQ}" ]]; then
  echo "missing requirements file: ${REQ}" >&2
  exit 1
fi

echo "python: ${PYTHON_BIN}"
"${PYTHON_BIN}" -c 'import sys; print("interpreter:", sys.executable); print("version:", sys.version)'
echo "requirements: ${REQ}"

# Install only the minimal test dependency, never the sglang meta-package.
"${PYTHON_BIN}" -m pip install --no-deps -r "${REQ}"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" -m pytest -q "${HERE}/test_dsa_topk_bf16.py" "$@"
