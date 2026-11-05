#!/usr/bin/env bash
# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Prepare an isolated python3.12 virtualenv for the extensible KV cache checks.
#
# The CI job previously ran a bare `pip install -e python`, which resolved
# wheels for whatever interpreter happened to be first on PATH (cp314 in the
# failing run) while the test steps activated a .venv that was never created.
# This script makes the interpreter explicit and fails fast when it is wrong.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

log() { printf '[setup-extensible-kv-cache] %s\n' "$*" >&2; }

log "repo root: ${REPO_ROOT}"
log "requested interpreter: ${PYTHON_BIN}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  log "ERROR: ${PYTHON_BIN} not found on PATH"
  log "PATH=${PATH}"
  exit 1
fi

log "which ${PYTHON_BIN}: $(command -v "${PYTHON_BIN}")"
log "${PYTHON_BIN} -V: $(${PYTHON_BIN} -V 2>&1)"

# Recreate the venv when it is missing or built from another interpreter.
if [ -d "${VENV_DIR}" ]; then
  if ! "${VENV_DIR}/bin/python" -c 'import sys; assert sys.version_info[:2]==(3,12), sys.version' >/dev/null 2>&1; then
    log "existing .venv is not python3.12, recreating"
    rm -rf "${VENV_DIR}"
  fi
fi

if [ ! -d "${VENV_DIR}" ]; then
  log "creating venv at ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# Guard against an inherited PYTHONPATH or VIRTUAL_ENV pointing elsewhere.
unset PYTHONPATH || true
export VIRTUAL_ENV="${VENV_DIR}"
export PATH="${VENV_DIR}/bin:${PATH}"

VENV_PY="${VENV_DIR}/bin/python"

log "which python: $(command -v python || true)"
log "${VENV_PY} -V: $(${VENV_PY} -V 2>&1)"

"${VENV_PY}" -c 'import sys; assert sys.version_info[:2]==(3,12), sys.version'

log "upgrading pip"
"${VENV_PY}" -m pip install --upgrade pip

log "installing sglang editable"
"${VENV_PY}" -m pip install -e "${REPO_ROOT}/python"

log "installing test dependencies"
"${VENV_PY}" -m pip install pytest pytest-randomly

log "verifying sglang import"
"${VENV_PY}" -c 'import sglang; print("sglang", sglang.__version__)'

log "environment ready: ${VENV_PY}"
