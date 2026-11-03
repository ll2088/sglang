#!/usr/bin/env bash
# Thin wrapper around the stdlib-only static checker. Kept as a separate
# entry point so CI can call it without knowing the Python module layout.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/check_static_patterns.py"
