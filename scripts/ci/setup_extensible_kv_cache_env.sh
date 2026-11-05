#!/usr/bin/env bash
# Single CI entrypoint for the extensible KV cache checks.
# Creates a python3.12 virtualenv, installs the project into it, and fails
# fast with diagnostics if the interpreter is not the expected one.
set -euxo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

VENV_DIR="${REPO_ROOT}/.venv"
PYTHON_BIN="python3.12"

log() { echo "[setup_extensible_kv_cache_env] $*"; }

log "repo root: ${REPO_ROOT}"
log "pwd: $(pwd)"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  log "ERROR: ${PYTHON_BIN} not found on PATH"
  command -v python3 || true
  python3 -V || true
  exit 1
fi

log "which ${PYTHON_BIN}: $(command -v "${PYTHON_BIN}")"
"${PYTHON_BIN}" -V

if [ ! -x "${VENV_DIR}/bin/python" ]; then
  log "creating virtualenv at ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

VENV_PY="${VENV_DIR}/bin/python"
log "venv python: ${VENV_PY}"
"${VENV_PY}" -V

# Assert the venv interpreter is python3.12 and that sys.prefix points at the
# venv, so pip never falls back to a system interpreter (cp314 wheels).
"${VENV_PY}" - <<'PY'
import sys
assert sys.version_info[:2] == (3, 12), sys.version
assert sys.prefix.endswith(".venv"), sys.prefix
print("interpreter:", sys.executable)
print("version:", sys.version)
print("prefix:", sys.prefix)
PY

log "upgrading pip inside venv"
"${VENV_PY}" -m pip install --upgrade pip

log "installing project into venv"
"${VENV_PY}" -m pip install -e "${REPO_ROOT}/python"

log "pip check"
"${VENV_PY}" -m pip check

log "import smoke test"
"${VENV_PY}" -c 'import sglang; print("sglang import ok")'

log "environment ready"
