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
# Prepare a Python 3.12 virtualenv for the extensible KV cache tests.
#
# The CI image may ship a newer system Python (e.g. 3.14) whose pip resolves
# cp314 wheels. This script pins every step to python3.12 and fails loudly if
# the interpreter or the resolved wheel tags drift.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
LOG_FILE="${REPO_ROOT}/.venv-setup.log"

on_error() {
  local exit_code=$?
  echo "[setup_extensible_kv_cache_env] failed with exit code ${exit_code}" >&2
  echo "[setup_extensible_kv_cache_env] last 100 log lines:" >&2
  tail -n 100 "${LOG_FILE}" >&2 || true
  exit "${exit_code}"
}
trap on_error ERR

log() {
  echo "[setup_extensible_kv_cache_env] $*" | tee -a "${LOG_FILE}"
}

: > "${LOG_FILE}"

if ! command -v python3.12 >/dev/null 2>&1; then
  log "python3.12 not found on PATH"
  exit 1
fi

log "creating venv at ${VENV_DIR}"
rm -rf "${VENV_DIR}"
python3.12 -m venv "${VENV_DIR}" 2>&1 | tee -a "${LOG_FILE}"

VENV_PY="${VENV_DIR}/bin/python"
if [ ! -x "${VENV_PY}" ]; then
  log "venv python missing at ${VENV_PY}"
  exit 1
fi

log "verifying interpreter version"
"${VENV_PY}" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' 2>&1 | tee -a "${LOG_FILE}"

log "pip version before install"
"${VENV_PY}" -m pip --version 2>&1 | tee -a "${LOG_FILE}"

log "pip compatibility tags before install"
"${VENV_PY}" -m pip debug --verbose 2>&1 | tee -a "${LOG_FILE}"
if "${VENV_PY}" -m pip debug --verbose 2>&1 | grep -q 'cp314'; then
  log "pip resolved cp314 tags; refusing to continue"
  exit 1
fi

log "upgrading pip/setuptools/wheel"
"${VENV_PY}" -m pip install --upgrade pip setuptools wheel 2>&1 | tee -a "${LOG_FILE}"

log "installing project in editable mode"
"${VENV_PY}" -m pip install -e "${REPO_ROOT}/python" 2>&1 | tee -a "${LOG_FILE}"

log "pip version after install"
"${VENV_PY}" -m pip --version 2>&1 | tee -a "${LOG_FILE}"

log "pip compatibility tags after install"
"${VENV_PY}" -m pip debug --verbose 2>&1 | tee -a "${LOG_FILE}"
if "${VENV_PY}" -m pip debug --verbose 2>&1 | grep -q 'cp314'; then
  log "pip resolved cp314 tags after install; refusing to continue"
  exit 1
fi

log "environment ready"
