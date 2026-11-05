#!/usr/bin/env bash
# Prepare a Python 3.12 virtualenv for the extensible KV cache tests.
#
# The script is intentionally strict: every install goes through
# .venv/bin/python -m pip, the interpreter is asserted to live inside the
# venv, and any cp314 wheel in the install log aborts the run. This keeps
# repeated CI runs idempotent and prevents leaking into the system
# dist-packages.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/.ci-logs}"
mkdir -p "$LOG_DIR"
PIP_LOG="$LOG_DIR/pip-install.log"

log() { printf '[setup-extensible-kv-cache] %s\n' "$*" >&2; }

fail() {
  log "ERROR: $*"
  log "python: $(command -v python || true)"
  log "pip: $(command -v pip || true)"
  if [ -x "$VENV_DIR/bin/python" ]; then
    "$VENV_DIR/bin/python" -c 'import sys; print("sys.executable:", sys.executable)' >&2 || true
  fi
  exit 1
}

# 1) Create the venv with an explicit 3.12 interpreter.
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  fail "$PYTHON_BIN not found on PATH"
fi
if [ ! -x "$VENV_DIR/bin/python" ]; then
  log "creating venv at $VENV_DIR with $PYTHON_BIN"
  "$PYTHON_BIN" -m venv "$VENV_DIR" || fail "venv creation failed"
fi

# 2) Activate and assert the interpreter/pip resolve inside the venv.
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

PYTHON_PATH="$(command -v python || true)"
PIP_PATH="$(command -v pip || true)"
case "$PYTHON_PATH" in
  "$VENV_DIR"/*) ;;
  *) fail "python resolves outside venv: ${PYTHON_PATH:-<none>}" ;;
esac
case "$PIP_PATH" in
  "$VENV_DIR"/*) ;;
  *) fail "pip resolves outside venv: ${PIP_PATH:-<none>}" ;;
esac

SYS_EXEC="$(python -c 'import sys; print(sys.executable)')"
case "$SYS_EXEC" in
  "$VENV_DIR"/*) ;;
  *) fail "sys.executable outside venv: $SYS_EXEC" ;;
esac

python -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' \
  || fail "venv interpreter is not Python 3.12"

# 3) Pin pip behaviour: no system fallback, no cp314 wheels, no cache noise.
export PIP_REQUIRE_VIRTUALENV=true
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_CACHE_DIR=1
export PIP_ONLY_BINARY=":all:"

# 4) Install through the venv interpreter only.
log "installing project into $VENV_DIR"
: > "$PIP_LOG"
python -m pip install --upgrade pip >>"$PIP_LOG" 2>&1 \
  || fail "pip self-upgrade failed (see $PIP_LOG)"
python -m pip install --require-virtualenv -e python >>"$PIP_LOG" 2>&1 \
  || fail "project install failed (see $PIP_LOG)"

# 5) Reject any cp314 wheel that slipped through.
if grep -E 'cp314' "$PIP_LOG" >/dev/null 2>&1; then
  grep -E 'cp314' "$PIP_LOG" >&2 || true
  fail "cp314 wheel detected in install log"
fi

# 6) Diagnostics for the CI log.
log "python --version: $(python --version 2>&1)"
log "pip --version: $(python -m pip --version 2>&1)"
log "sys.executable: $SYS_EXEC"
log "site-packages: $(python -c 'import site; print(site.getsitepackages()[0])')"
log "torch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo unavailable)"
log "pytest: $(python -m pytest --version 2>&1 | head -n1 || echo unavailable)"
log "done"
