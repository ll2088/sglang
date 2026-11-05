#!/usr/bin/env bash
# Prepare a python3.12 virtualenv for the extensible KV cache checks.
#
# The CI image ships several interpreters; a bare `pip install` can resolve
# wheels for the wrong one (e.g. cp314) and land in the system site-packages.
# This script pins the interpreter, asserts the venv is actually in use, and
# fails loudly with diagnostics instead of letting pytest run in a bad env.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"

log() { printf '[setup-extensible-kv-cache] %s\n' "$*" >&2; }

fail() {
    log "ERROR: $*"
    log "which python: $(command -v python || echo '<none>')"
    log "which python3.12: $(command -v python3.12 || echo '<none>')"
    log "which pip: $(command -v pip || echo '<none>')"
    exit 1
}

# 1. Resolve an absolute python3.12 path.
PYTHON_BIN="$(command -v python3.12 || true)"
if [[ -z "${PYTHON_BIN}" ]]; then
    for candidate in /usr/bin/python3.12 /usr/local/bin/python3.12; do
        if [[ -x "${candidate}" ]]; then
            PYTHON_BIN="${candidate}"
            break
        fi
    done
fi
[[ -n "${PYTHON_BIN}" ]] || fail "python3.12 not found"
log "using interpreter: ${PYTHON_BIN}"

# 2. Create the venv with that interpreter (idempotent).
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    log "creating venv at ${VENV_DIR}"
    "${PYTHON_BIN}" -m venv "${VENV_DIR}" || fail "venv creation failed"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# 3. Assert the venv interpreter is the one we expect, before installing.
assert_venv() {
    python - <<'PY' || exit 1
import os
import sys

venv = os.environ.get("VIRTUAL_ENV")
print("executable:", sys.executable)
print("version:", sys.version)
print("prefix:", sys.prefix)
print("VIRTUAL_ENV:", venv)
if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"expected python 3.12, got {sys.version_info[:3]}")
if not venv or os.path.realpath(sys.prefix) != os.path.realpath(venv):
    raise SystemExit(f"sys.prefix {sys.prefix!r} is not the venv {venv!r}")
PY
}

assert_venv || fail "venv interpreter assertion failed before install"

log "which python: $(command -v python)"
log "which pip: $(command -v pip || echo '<none>')"

# 4. Install with the venv interpreter, never a bare pip.
python -m pip --version || fail "pip missing in venv"
python -m pip install --upgrade pip || fail "pip upgrade failed"
python -m pip install -e "${REPO_ROOT}/python" || fail "editable install failed"

# 5. Re-assert after install: some packages can shadow the interpreter.
assert_venv || fail "venv interpreter assertion failed after install"

log "environment ready"
