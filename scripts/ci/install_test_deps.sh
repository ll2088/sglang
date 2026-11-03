#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Minimal, reproducible test dependency installer for the ported regression
# tests. It intentionally avoids resolving sglang's full dependency tree.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
EXPECTED_PYTHON="${EXPECTED_PYTHON:-3.12}"

actual_version="$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$actual_version" != "$EXPECTED_PYTHON" ]]; then
  echo "ERROR: expected Python ${EXPECTED_PYTHON}, got ${actual_version}" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONSTRAINTS="${SCRIPT_DIR}/constraints-ci.txt"
REQUIREMENTS="${SCRIPT_DIR}/requirements-test-min.txt"

if [[ ! -f "$CONSTRAINTS" ]]; then
  echo "ERROR: missing constraints file: $CONSTRAINTS" >&2
  exit 1
fi
if [[ ! -f "$REQUIREMENTS" ]]; then
  echo "ERROR: missing requirements file: $REQUIREMENTS" >&2
  exit 1
fi

# Install only the minimal test dependencies. --no-deps prevents pip from
# pulling in sglang's full runtime dependency set (py-spy, pyzmq, ...).
"$PYTHON_BIN" -m pip install --no-deps \
  -c "$CONSTRAINTS" \
  -r "$REQUIREMENTS"

# Install the local package without dependencies so imports resolve to the
# working tree, but no heavy runtime deps are pulled in.
"$PYTHON_BIN" -m pip install --no-deps -e "${REPO_ROOT}/python"

echo "Test dependencies installed for Python ${actual_version}"
