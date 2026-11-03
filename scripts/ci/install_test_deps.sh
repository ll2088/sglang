#!/usr/bin/env bash
# Install the minimal dependency set required by the quant regression tests.
# This script must NOT run `pip install .` / `pip install -e .`, because that
# would resolve the full sglang dependency graph and is unnecessary for the
# self-contained tests under test/registered/quant/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

REQ_FILE="${SCRIPT_DIR}/requirements-test-min.txt"
CONSTRAINTS_FILE="${SCRIPT_DIR}/constraints-ci.txt"

if [[ ! -f "${REQ_FILE}" ]]; then
  echo "[install_test_deps] missing requirements file: ${REQ_FILE}" >&2
  exit 1
fi

if [[ ! -f "${CONSTRAINTS_FILE}" ]]; then
  echo "[install_test_deps] missing constraints file: ${CONSTRAINTS_FILE}" >&2
  exit 1
fi

echo "[install_test_deps] repo root: ${REPO_ROOT}"
echo "[install_test_deps] installing: ${REQ_FILE} (constraints: ${CONSTRAINTS_FILE})"

if ! python -m pip install --disable-pip-version-check \
    -r "${REQ_FILE}" -c "${CONSTRAINTS_FILE}"; then
  echo "[install_test_deps] pip install failed" >&2
  exit 1
fi

echo "[install_test_deps] done"
