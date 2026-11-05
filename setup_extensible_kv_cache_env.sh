#!/usr/bin/env bash
# Prepare an isolated Python 3.12 environment for the extensible KV cache
# tests. The script is intentionally strict: any mismatch between the
# interpreter that creates the venv and the one that installs/runs the tests
# aborts immediately, because a system Python (e.g. 3.14) would otherwise
# resolve cp314 wheels and break the 3.12 assertions in the test commands.
set -euo pipefail

# Never inherit a caller's Python search path or home; both can make pip and
# the interpreter pick up system site-packages instead of the venv.
unset PYTHONPATH || true
unset PYTHONHOME || true
unset VIRTUAL_ENV || true

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
PYTHON_BIN="${PYTHON312_BIN:-python3.12}"

log() { printf '[setup_extensible_kv_cache_env] %s\n' "$*" >&2; }

fail() {
    log "ERROR: $*"
    exit 1
}

# 1) Locate an absolute Python 3.12 interpreter and verify it before use.
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    fail "${PYTHON_BIN} not found; install Python 3.12 or set PYTHON312_BIN"
fi
PYTHON_ABS="$(command -v "${PYTHON_BIN}")"
case "${PYTHON_ABS}" in
    /*) ;;
    *) fail "resolved interpreter is not an absolute path: ${PYTHON_ABS}" ;;
esac
PY_VERSION="$("${PYTHON_ABS}" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
if [ "${PY_VERSION}" != "3.12" ]; then
    fail "${PYTHON_ABS} reports Python ${PY_VERSION}, expected 3.12"
fi
log "using interpreter ${PYTHON_ABS} (Python ${PY_VERSION})"

# 2) Recreate the venv from that exact interpreter.
if [ -d "${VENV_DIR}" ]; then
    log "removing existing venv at ${VENV_DIR}"
    rm -rf "${VENV_DIR}"
fi
"${PYTHON_ABS}" -m venv "${VENV_DIR}" || fail "failed to create venv"
VENV_PYTHON="${VENV_DIR}/bin/python"
[ -x "${VENV_PYTHON}" ] || fail "venv python missing at ${VENV_PYTHON}"

# 3) Validate the venv interpreter immediately after creation.
VENV_VERSION="$("${VENV_PYTHON}" -V 2>&1)"
log "venv interpreter: ${VENV_VERSION}"
case "${VENV_VERSION}" in
    "Python 3.12."*) ;;
    *) fail "venv interpreter is not 3.12.x: ${VENV_VERSION}" ;;
esac

# 4) Install with the venv interpreter explicitly, never a bare pip.
log "upgrading pip/setuptools/wheel in venv"
"${VENV_PYTHON}" -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    || fail "failed to upgrade pip tooling"

log "installing project in editable mode"
"${VENV_PYTHON}" -m pip install --no-cache-dir -e "${REPO_ROOT}/python" \
    || fail "failed to install project"

# 5) Post-install smoke checks: version and key imports.
"${VENV_PYTHON}" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' \
    || fail "post-install interpreter is not Python 3.12"

"${VENV_PYTHON}" - <<'PY' || fail "post-install import smoke check failed"
import importlib
import sys

assert sys.version_info[:2] == (3, 12), sys.version
for name in ("torch", "sglang"):
    importlib.import_module(name)
print("import smoke check passed", file=sys.stderr)
PY

log "environment ready: ${VENV_DIR}"
