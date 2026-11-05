#!/usr/bin/env bash
# Prepare a Python 3.12 virtualenv for the extensible KV cache checks.
#
# The previous CI setup reused the system interpreter, which resolved
# cp314 wheels and failed before pytest ever ran. This script pins the
# interpreter, verifies it, and keeps the full pip log on disk so a
# failure can be diagnosed from the CI artifact.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
LOG_DIR="${REPO_ROOT}/ci-logs"
PIP_LOG="${LOG_DIR}/pip-install.log"
PYTHON_BIN="${PYTHON312_BIN:-python3.12}"

mkdir -p "${LOG_DIR}"

log() { printf '[setup] %s\n' "$*" >&2; }

fail() {
  log "ERROR: $*"
  log "which ${PYTHON_BIN}: $(command -v "${PYTHON_BIN}" || echo '<not found>')"
  log "${PYTHON_BIN} -V: $("${PYTHON_BIN}" -V 2>&1 || echo '<failed>')"
  log "PATH: ${PATH}"
  exit 1
}

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  fail "${PYTHON_BIN} not found; install Python 3.12 or set PYTHON312_BIN"
fi

log "using interpreter: $(command -v "${PYTHON_BIN}")"
log "interpreter version: $("${PYTHON_BIN}" -V 2>&1)"

if [ ! -x "${VENV_DIR}/bin/python" ]; then
  log "creating venv at ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}" || fail "venv creation failed"
fi

VENV_PY="${VENV_DIR}/bin/python"
if ! "${VENV_PY}" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' >/dev/null 2>&1; then
  fail "venv interpreter is not Python 3.12: $("${VENV_PY}" -V 2>&1)"
fi
log "venv interpreter verified: $("${VENV_PY}" -V 2>&1)"

# Upgrade pip inside the venv only, so the system interpreter is never used.
"${VENV_PY}" -m pip install --upgrade pip setuptools wheel \
  >>"${PIP_LOG}" 2>&1 || fail "pip bootstrap failed; see ${PIP_LOG}"

# Install the project in editable mode. --only-binary=:all: keeps pip from
# building sdists against the wrong interpreter, and the log is kept whole.
set -x
"${VENV_PY}" -m pip install --only-binary=:all: -e "${REPO_ROOT}/python" \
  >>"${PIP_LOG}" 2>&1 || {
  set +x
  fail "pip install failed; see ${PIP_LOG}"
}
set +x

log "environment ready; pip log at ${PIP_LOG}"
