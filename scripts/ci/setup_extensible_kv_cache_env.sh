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
# Prepare an isolated python3.12 virtualenv for the extensible KV cache tests.
#
# Every install and test command must go through .venv/bin/python so that the
# system interpreter (and its site-packages) is never used. The script fails
# loudly with full context instead of silently falling back.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

log() {
    printf '[setup_extensible_kv_cache_env] %s\n' "$*" >&2
}

fail() {
    log "ERROR: $*"
    exit 1
}

log "repo root: ${REPO_ROOT}"
log "requested interpreter: ${PYTHON_BIN}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    fail "${PYTHON_BIN} not found on PATH; install python3.12 before running this script"
fi

log "which ${PYTHON_BIN}: $(command -v "${PYTHON_BIN}")"
log "${PYTHON_BIN} -V: $("${PYTHON_BIN}" -V 2>&1)"

if ! "${PYTHON_BIN}" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' >/dev/null 2>&1; then
    fail "${PYTHON_BIN} is not python3.12: $("${PYTHON_BIN}" -V 2>&1)"
fi

if [ ! -x "${VENV_DIR}/bin/python" ]; then
    log "creating virtualenv at ${VENV_DIR}"
    "${PYTHON_BIN}" -m venv "${VENV_DIR}" || fail "failed to create virtualenv at ${VENV_DIR}"
fi

VENV_PYTHON="${VENV_DIR}/bin/python"

if [ ! -x "${VENV_PYTHON}" ]; then
    fail "virtualenv python missing: ${VENV_PYTHON}"
fi

log "venv python: ${VENV_PYTHON}"
log "venv python -V: $("${VENV_PYTHON}" -V 2>&1)"

if ! "${VENV_PYTHON}" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' >/dev/null 2>&1; then
    fail "virtualenv interpreter is not python3.12: $("${VENV_PYTHON}" -V 2>&1)"
fi

# Make sure the venv interpreter does not see the system site-packages.
if "${VENV_PYTHON}" -c 'import sys; print("\n".join(sys.path))' | grep -q '/usr/lib/python3/dist-packages'; then
    fail "virtualenv still sees /usr/lib/python3/dist-packages; isolation failed"
fi

log "upgrading pip inside the virtualenv"
"${VENV_PYTHON}" -m pip install --upgrade pip || fail "failed to upgrade pip inside the virtualenv"

log "pip version: $("${VENV_PYTHON}" -m pip --version 2>&1)"
log "pip debug --verbose (compatible tags):"
"${VENV_PYTHON}" -m pip debug --verbose 2>&1 | sed -n '1,40p' >&2 || true

# Reject cp314 wheels: the venv is python3.12, so any cp314 wheel means pip
# resolved against the wrong interpreter or a stale cache.
if "${VENV_PYTHON}" -m pip debug --verbose 2>&1 | grep -q 'cp314'; then
    fail "pip reports cp314 compatibility tags inside a python3.12 virtualenv"
fi

log "installing the project in editable mode inside the virtualenv"
"${VENV_PYTHON}" -m pip install -e "${REPO_ROOT}/python" || fail "failed to install the project inside the virtualenv"

log "verifying installed packages do not come from the system site-packages"
if "${VENV_PYTHON}" -m pip list --format=freeze 2>/dev/null | grep -q '^requests=='; then
    REQ_LOCATION="$("${VENV_PYTHON}" -c 'import requests, os; print(os.path.dirname(requests.__file__))' 2>/dev/null || true)"
    if [ -n "${REQ_LOCATION}" ] && printf '%s' "${REQ_LOCATION}" | grep -q '/usr/lib/python3/dist-packages'; then
        fail "requests resolved from the system site-packages: ${REQ_LOCATION}"
    fi
fi

log "environment ready: ${VENV_PYTHON}"
