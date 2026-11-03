#!/usr/bin/env bash
# Install the minimal dependency set used by the quant regression tests.
# Deliberately does NOT install the sglang package itself, so that the
# tests cannot accidentally depend on the heavy runtime dependency tree.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ_FILE="${SCRIPT_DIR}/requirements-test-min.txt"

if [[ ! -f "${REQ_FILE}" ]]; then
  echo "ERROR: requirements file not found: ${REQ_FILE}" >&2
  exit 1
fi

echo "Installing minimal test dependencies from ${REQ_FILE}"
python -m pip install --upgrade pip >/dev/null
if ! python -m pip install -r "${REQ_FILE}"; then
  echo "ERROR: pip install failed for ${REQ_FILE}" >&2
  exit 1
fi

echo "Minimal test dependencies installed successfully"
