#!/usr/bin/env bash
# Prepare an isolated Python 3.12 environment for the extensible KV cache
# tests. The script must never fall back to the system interpreter: a stray
# system pip resolves wheels for the host Python (e.g. cp314) and installs
# outside the venv, which is what broke the previous CI run.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

log() { printf '[setup_extensible_kv_cache_env] %s\n' "$*" >&2; }

fail() {
    log "ERROR: $*"
    exit 1
}

# Drop anything that could leak the system interpreter or its site-packages
# into the venv (PYTHONHOME in particular makes venv creation misbehave).
unset PYTHONPATH || true
unset PYTHONHOME || true
unset VIRTUAL_ENV || true

command -v "${PYTHON_BIN}" >/dev/null 2>&1 \
    || fail "${PYTHON_BIN} not found; install Python 3.12 or set PYTHON_BIN"

log "Using interpreter: $(command -v "${PYTHON_BIN}")"
"${PYTHON_BIN}" -V

# Recreate the venv from scratch so a stale or partially populated one from a
# previous attempt cannot be reused.
if [ -d "${VENV_DIR}" ]; then
    log "Removing existing venv at ${VENV_DIR}"
    rm -rf "${VENV_DIR}"
fi

log "Creating venv at ${VENV_DIR}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}" \
    || fail "failed to create venv with ${PYTHON_BIN}"

VENV_PYTHON="${VENV_DIR}/bin/python"
[ -x "${VENV_PYTHON}" ] || fail "venv python missing at ${VENV_PYTHON}"

# Assert the venv really is 3.12 before installing anything.
VENV_VERSION="$("${VENV_PYTHON}" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
log "venv python version: ${VENV_VERSION}"
[ "${VENV_VERSION}" = "3.12" ] \
    || fail "venv interpreter is ${VENV_VERSION}, expected 3.12"

# Make sure the venv does not see the system site-packages.
"${VENV_PYTHON}" -c 'import sys; assert not any("dist-packages" in p for p in sys.path), sys.path'

log "Upgrading pip inside the venv"
"${VENV_PYTHON}" -m pip install --upgrade pip \
    || fail "failed to upgrade pip inside the venv"

"${VENV_PYTHON}" -m pip -V

log "Installing sglang (editable) into the venv"
if ! "${VENV_PYTHON}" -m pip install --verbose -e "${REPO_ROOT}/python"; then
    fail "pip install failed; see verbose log above"
fi

log "Smoke check: import sglang"
"${VENV_PYTHON}" -c 'import sglang; print("sglang imported from", sglang.__file__)' \
    || fail "import sglang failed inside the venv"

log "Environment ready: ${VENV_DIR}"
