#!/usr/bin/env bash
# Install only the minimal dependencies needed by the regression tests.
# This avoids resolving sglang's full runtime dependency tree in CI.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ_FILE="${SCRIPT_DIR}/requirements-test-min.txt"

python -m pip install --upgrade pip
python -m pip install -r "${REQ_FILE}"
