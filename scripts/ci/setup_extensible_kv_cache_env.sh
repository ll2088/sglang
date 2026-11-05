#!/usr/bin/env bash
# Prepare the environment used by the extensible KV cache verification jobs.
#
# The verification commands require Python 3.12. CI images may ship a newer
# interpreter (e.g. 3.14), and pip would then resolve cp314 wheels, which is
# not what the tests are validated against. This script therefore:
#   1. fails fast on any error and prints diagnostics,
#   2. locates a Python 3.12 interpreter explicitly,
#   3. creates/uses a venv from that interpreter,
#   4. installs dependencies with that venv's pip only.
set -euo pipefail

log() { printf '[setup_extensible_kv_cache_env] %s\n' "$*" >&2; }

on_error() {
  local exit_code=$?
  log "environment setup failed with exit code ${exit_code}"
  log "python candidates on PATH:"
  command -v python3.12 || true
  command -v python3 || true
  python3 --version || true
  exit "${exit_code}"
}
trap on_error ERR

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# 1. Locate Python 3.12 explicitly. Do not fall back to whatever `python3`
#    happens to be: a 3.14 interpreter silently changes wheel selection.
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  for candidate in python3.12 python3; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      if "${candidate}" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)' 2>/dev/null; then
        PYTHON_BIN="$(command -v "${candidate}")"
        break
      fi
    fi
  done
fi

if [[ -z "${PYTHON_BIN}" ]]; then
  log "no Python 3.12 interpreter found; install one (e.g. actions/setup-python with 3.12) or set PYTHON_BIN"
  exit 1
fi

log "using interpreter: ${PYTHON_BIN}"
"${PYTHON_BIN}" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'

# 2. Create or reuse a venv from that interpreter.
VENV_DIR="${VENV_DIR:-.venv}"
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  log "creating venv at ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

VENV_PYTHON="${VENV_DIR}/bin/python"
"${VENV_PYTHON}" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
log "venv interpreter: $("${VENV_PYTHON}" -c 'import sys; print(sys.executable, sys.version.split()[0])')"

# 3. Install with the venv's pip only, so wheel tags match Python 3.12.
"${VENV_PYTHON}" -m pip install --upgrade pip

# Prefer wheels to avoid long source builds in CI; fall back to source only
# when a wheel is genuinely unavailable for this platform.
if ! "${VENV_PYTHON}" -m pip install --only-binary=:all: -e '.[dev]'; then
  log "binary-only install failed; retrying allowing source builds"
  "${VENV_PYTHON}" -m pip install -e '.[dev]'
fi

log "environment ready"
