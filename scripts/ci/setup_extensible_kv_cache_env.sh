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
# The CI environment must call this script instead of a bare `pip install`:
# the system interpreter may be newer than 3.12 and resolve wheels for the
# wrong ABI tag (e.g. cp314), which then fails at import time.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
VENV_DIR="${VENV_DIR:-.venv}"

log() { printf '[setup_extensible_kv_cache_env] %s\n' "$*" >&2; }

fail() {
  log "ERROR: $*"
  log "--- diagnostics ---"
  command -v "${PYTHON_BIN}" || true
  "${PYTHON_BIN}" -V 2>&1 || true
  if [ -x "${VENV_DIR}/bin/python" ]; then
    "${VENV_DIR}/bin/python" -V 2>&1 || true
    "${VENV_DIR}/bin/python" -m pip debug --verbose 2>&1 | head -n 40 || true
  fi
  exit 1
}

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  fail "${PYTHON_BIN} not found on PATH"
fi

log "Using interpreter: $(command -v "${PYTHON_BIN}")"
"${PYTHON_BIN}" -V

if [ ! -x "${VENV_DIR}/bin/python" ]; then
  log "Creating virtualenv at ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}" || fail "failed to create virtualenv"
fi

VENV_PY="${VENV_DIR}/bin/python"

# Hard gate: the venv must be 3.12 before any dependency is resolved.
if ! "${VENV_PY}" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' 2>/dev/null; then
  fail "virtualenv interpreter is not Python 3.12"
fi
log "Virtualenv interpreter: $("${VENV_PY}" -V 2>&1)"

# Never resolve wheels for another ABI tag.
export PIP_ONLY_BINARY="${PIP_ONLY_BINARY:-:all:}"
export PIP_DISABLE_PIP_VERSION_CHECK=1

log "Upgrading pip/setuptools/wheel"
"${VENV_PY}" -m pip install --require-virtualenv --only-binary=:all: \
  --upgrade pip setuptools wheel || fail "failed to bootstrap pip"

log "Installing sglang (editable) into ${VENV_DIR}"
"${VENV_PY}" -m pip install --require-virtualenv --only-binary=:all: \
  -e python || fail "failed to install sglang"

log "Installing test dependencies"
"${VENV_PY}" -m pip install --require-virtualenv --only-binary=:all: \
  pytest pytest-randomly || fail "failed to install test dependencies"

# Final sanity check: imports must work under the pinned interpreter.
"${VENV_PY}" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' \
  || fail "post-install interpreter check failed"

log "Environment ready. Activate with: . ${VENV_DIR}/bin/activate"
