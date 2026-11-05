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
# Prepare an isolated Python 3.12 environment for the extensible KV cache
# checks. Every install and test invocation goes through .venv/bin/python so
# that a system pip (which may resolve cp314 wheels) can never be used.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

log() { printf '[setup_extensible_kv_cache_env] %s\n' "$*" >&2; }

on_error() {
  local status=$?
  log "environment setup failed with exit code ${status}"
  log "which ${PYTHON_BIN}: $(command -v "${PYTHON_BIN}" || echo '<not found>')"
  log "${PYTHON_BIN} -V: $("${PYTHON_BIN}" -V 2>&1 || echo '<failed>')"
  log "PATH: ${PATH}"
  if [[ -x "${VENV_DIR}/bin/python" ]]; then
    log "venv python: $("${VENV_DIR}/bin/python" -V 2>&1 || echo '<failed>')"
    log "venv pip: $("${VENV_DIR}/bin/python" -m pip -V 2>&1 || echo '<failed>')"
  fi
  exit "${status}"
}
trap on_error ERR

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  log "${PYTHON_BIN} not found on PATH; install Python 3.12 or set PYTHON_BIN"
  exit 1
fi

log "creating venv at ${VENV_DIR} with ${PYTHON_BIN}"
rm -rf "${VENV_DIR}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"

VENV_PY="${VENV_DIR}/bin/python"
if [[ ! -x "${VENV_PY}" ]]; then
  log "venv python missing at ${VENV_PY}"
  exit 1
fi

log "venv python: $("${VENV_PY}" -V)"
log "venv pip: $("${VENV_PY}" -m pip -V)"
log "venv sys.executable: $("${VENV_PY}" -c 'import sys; print(sys.executable)')"

"${VENV_PY}" - <<'PY'
import sys
assert sys.version_info[:2] == (3, 12), sys.version
print(f"interpreter ok: {sys.executable} {sys.version.split()[0]}")
PY

log "upgrading pip inside venv"
"${VENV_PY}" -m pip install --upgrade pip

log "installing sglang editable into venv"
"${VENV_PY}" -m pip install -e "${REPO_ROOT}/python"

log "environment ready"
