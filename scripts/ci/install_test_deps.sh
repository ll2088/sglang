#!/usr/bin/env bash
# Install the minimal dependency set used by the targeted regression tests.
# The sglang package itself is installed with --no-deps so that CI does not
# resolve the full runtime dependency graph during environment preparation.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REQ_FILE="${ROOT_DIR}/scripts/ci/requirements-test-min.txt"
CONSTRAINTS_FILE="${ROOT_DIR}/scripts/ci/constraints-ci.txt"

if [[ ! -f "${REQ_FILE}" ]]; then
  echo "missing requirements file: ${REQ_FILE}" >&2
  exit 1
fi
if [[ ! -f "${CONSTRAINTS_FILE}" ]]; then
  echo "missing constraints file: ${CONSTRAINTS_FILE}" >&2
  exit 1
fi

echo "[ci] installing sglang without dependencies"
python -m pip install --no-deps -e "${ROOT_DIR}/python"

echo "[ci] installing minimal test dependencies from ${REQ_FILE}"
python -m pip install -r "${REQ_FILE}" -c "${CONSTRAINTS_FILE}"

echo "[ci] installed packages:"
python -m pip list --format=columns | grep -E '^(pytest|pytest-randomly|sglang)' || true
