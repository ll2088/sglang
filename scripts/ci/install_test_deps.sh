#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Install minimal test dependencies without resolving the full sglang
# dependency tree. The local package is installed with --no-deps so that
# pip does not try to fetch py-spy, pybase64, pydantic, pyzmq, etc.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONSTRAINTS="${SCRIPT_DIR}/constraints-ci.txt"
REQ_TEST="${REPO_ROOT}/requirements-test-min.txt"

if [[ ! -f "${CONSTRAINTS}" ]]; then
  echo "[install_test_deps] missing constraints file: ${CONSTRAINTS}" >&2
  exit 1
fi

if [[ ! -f "${REQ_TEST}" ]]; then
  echo "[install_test_deps] missing test requirements: ${REQ_TEST}" >&2
  exit 1
fi

echo "[install_test_deps] upgrading pip/setuptools/wheel"
python -m pip install --upgrade pip setuptools wheel

echo "[install_test_deps] installing minimal test requirements"
python -m pip install -r "${REQ_TEST}" -c "${CONSTRAINTS}"

echo "[install_test_deps] installing local sglang with --no-deps"
python -m pip install --no-deps -c "${CONSTRAINTS}" -e "${REPO_ROOT}/python"

echo "[install_test_deps] done"
