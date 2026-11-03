#!/usr/bin/env bash
# Thin wrapper around check_static_patterns.py so CI can call a single script.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "=== static pattern review ==="
"${PYTHON_BIN}" "${SCRIPT_DIR}/check_static_patterns.py"
