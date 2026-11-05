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
# Prepare a reproducible Python 3.12 virtualenv for the extensible KV cache
# tests. The script is intentionally strict: it fails loudly instead of
# silently reusing a stale venv or the system interpreter, which is what made
# earlier CI runs install cp314 wheels and touch /usr/lib/python3/dist-packages.

set -euxo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

log() { echo "[setup_extensible_kv_cache_env] $*" >&2; }

fail() {
  log "ERROR: $*"
  if [ -x "${VENV_DIR}/bin/python" ]; then
    "${VENV_DIR}/bin/python" -m pip list || true
    "${VENV_DIR}/bin/python" -c 'import sys; print(sys.executable); print(sys.version); print("\n".join(sys.path))' || true
  fi
  exit 1
}

# 1. Locate the interpreter explicitly. Never rely on `python` from PATH.
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  fail "${PYTHON_BIN} not found on PATH; install Python 3.12 first"
fi
PYTHON_PATH="$(command -v "${PYTHON_BIN}")"
log "which ${PYTHON_BIN}: ${PYTHON_PATH}"
"${PYTHON_PATH}" -V
"${PYTHON_PATH}" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'

# 2. Recreate the venv from scratch so a stale interpreter cannot be reused.
rm -rf "${VENV_DIR}"
"${PYTHON_PATH}" -m venv --clear "${VENV_DIR}"

# 3. Activate and scrub inherited Python environment variables.
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP || true
export PIP_REQUIRE_VIRTUALENV=1
export PIP_NO_USER_CONFIG=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_CACHE_DIR=1

VENV_PY="${VENV_DIR}/bin/python"
log "venv python: ${VENV_PY}"
"${VENV_PY}" -V
"${VENV_PY}" -c 'import sys; print(sys.executable, sys.version)'
"${VENV_PY}" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'

# 4. Always install through the venv interpreter, never bare `pip`.
"${VENV_PY}" -m pip -V
"${VENV_PY}" -m pip install --upgrade pip
"${VENV_PY}" -m pip install -e "${REPO_ROOT}/python"

# 5. Assert the interpreter and key packages are importable before running tests.
"${VENV_PY}" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
"${VENV_PY}" -c 'import torch, pytest; print("torch", torch.__version__); print("pytest", pytest.__version__)'

log "environment ready: ${VENV_PY}"
