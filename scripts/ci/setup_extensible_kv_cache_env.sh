#!/usr/bin/env bash
# Prepare a Python 3.12 environment for the extensible KV cache checks.
#
# The CI runner may default to a newer interpreter (e.g. 3.14), which makes
# pip resolve cp314 wheels and breaks the pinned 3.12 validation commands.
# This script creates a fresh venv with an explicit 3.12 interpreter, asserts
# the version, and installs the project with that interpreter only.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

log() { printf '[setup-extensible-kv-cache] %s\n' "$*" >&2; }

find_python312() {
  for candidate in python3.12 python3 python; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      if "${candidate}" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)' 2>/dev/null; then
        command -v "${candidate}"
        return 0
      fi
    fi
  done
  return 1
}

if ! PYTHON312="$(find_python312)"; then
  log "error: no Python 3.12 interpreter found on PATH"
  log "PATH=${PATH}"
  for candidate in python3.14 python3.13 python3.12 python3 python; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      log "  ${candidate} -> $("${candidate}" -V 2>&1 || true)"
    fi
  done
  exit 1
fi

log "using interpreter: ${PYTHON312}"
"${PYTHON312}" -V

# Recreate the venv so a stale cp314 environment is never reused.
if [ -d "${VENV_DIR}" ]; then
  log "removing stale venv at ${VENV_DIR}"
  rm -rf "${VENV_DIR}"
fi

"${PYTHON312}" -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
. "${VENV_DIR}/bin/activate"

log "which python: $(command -v python)"
python -V
python -m pip -V

python -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'

python -m pip install --upgrade pip
# Install with the venv interpreter so wheels match cp312.
python -m pip install -e "${ROOT_DIR}/python"

log "environment ready"
