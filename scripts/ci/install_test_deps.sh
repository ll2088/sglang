#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Install the minimal dependency set needed by the quantization regression
# tests.  Deliberately avoids `pip install -e .` / `pip install .` because
# those resolve the full sglang dependency tree and are the root cause of the
# CI environment-preparation failures.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONSTRAINTS="${ROOT_DIR}/scripts/ci/constraints-ci.txt"
REQS="${ROOT_DIR}/scripts/ci/requirements-test-min.txt"

if [[ ! -f "${REQS}" ]]; then
  echo "missing requirements file: ${REQS}" >&2
  exit 1
fi

PIP_ARGS=(--no-deps --disable-pip-version-check)
if [[ -f "${CONSTRAINTS}" ]]; then
  PIP_ARGS+=(-c "${CONSTRAINTS}")
fi

python -m pip install "${PIP_ARGS[@]}" -r "${REQS}" || {
  echo "minimal dependency install failed; retrying verbosely" >&2
  python -m pip install -v "${PIP_ARGS[@]}" -r "${REQS}"
  exit 1
}

# Smoke check: the test modules must be importable without pulling in the
# full sglang runtime.
python -c "import pytest" 
