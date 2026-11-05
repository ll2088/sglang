#!/usr/bin/env bash
# Create and validate the Python 3.12 virtualenv used by CI checks.
# Fails loudly with interpreter diagnostics instead of silently falling back
# to the system Python (which may be 3.14 and pull cp314 wheels).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

log() { printf '[setup_env] %s\n' "$*" >&2; }

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  log "${PYTHON_BIN} not found on PATH"
  log "which python3: $(command -v python3 || true)"
  log "python3 -V: $(python3 -V 2>&1 || true)"
  exit 1
fi

if [ ! -x "${VENV_DIR}/bin/python" ]; then
  log "creating venv at ${VENV_DIR} with ${PYTHON_BIN}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
. "${VENV_DIR}/bin/activate"

if ! python -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' 2>/dev/null; then
  log "venv interpreter is not Python 3.12"
  log "which python: $(command -v python || true)"
  log "python -V: $(python -V 2>&1 || true)"
  log "pip -V: $(pip -V 2>&1 || true)"
  exit 1
fi

log "python: $(command -v python)"
log "python -V: $(python -V 2>&1)"
log "pip -V: $(pip -V 2>&1)"

# Keep the full pip output so dependency failures are diagnosable in CI logs.
if ! python -m pip install --upgrade pip; then
  log "pip self-upgrade failed"
  exit 1
fi

if ! python -m pip install -e "${REPO_ROOT}/python"; then
  log "editable install of python/ failed; see pip output above"
  exit 1
fi

log "environment ready"
