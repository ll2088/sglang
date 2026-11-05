#!/usr/bin/env bash
# Prepare a Python 3.12 virtualenv for the extensible KV cache checks.
#
# This script is meant to be sourced (or executed) by every CI dimension
# before running pytest, so that the interpreter used to install the project
# is the same one that runs the tests. It deliberately avoids a bare `pip`:
# on hosts where the system interpreter is newer (e.g. 3.14), `pip install -e`
# would resolve cp314 wheels and the later `. .venv/bin/activate` would not
# undo that.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

log() { printf '[setup_extensible_kv_cache_env] %s\n' "$*" >&2; }

log "repo root: ${REPO_ROOT}"
log "venv dir:  ${VENV_DIR}"
log "python:    ${PYTHON_BIN}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    log "error: ${PYTHON_BIN} not found on PATH"
    exit 1
fi

if [ ! -x "${VENV_DIR}/bin/python" ]; then
    log "creating virtualenv"
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
. "${VENV_DIR}/bin/activate"

log "python -V: $(python -V 2>&1)"
log "which python: $(command -v python)"
log "python -m pip -V: $(python -m pip -V 2>&1)"

if ! python -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' >/dev/null 2>&1; then
    log "error: activated interpreter is not Python 3.12: $(python -V 2>&1)"
    exit 1
fi

# Install through the activated interpreter only. `python -m pip` cannot fall
# back to the system pip even if PATH is odd.
python -m pip install --upgrade pip
python -m pip install -e "${REPO_ROOT}/python"

log "environment ready"
