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
# tests. Every install and check goes through .venv/bin/python so the system
# interpreter (and its dist-packages) is never used.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
PYTHON_BIN="${PYTHON312_BIN:-python3.12}"

log() { printf '[setup_extensible_kv_cache_env] %s\n' "$*" >&2; }
fail() { log "ERROR: $*"; exit 1; }

command -v "${PYTHON_BIN}" >/dev/null 2>&1 \
  || fail "${PYTHON_BIN} not found; install Python 3.12 or set PYTHON312_BIN"

"${PYTHON_BIN}" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' \
  || fail "${PYTHON_BIN} is not Python 3.12"

if [ ! -x "${VENV_DIR}/bin/python" ]; then
  log "creating venv at ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}" || fail "failed to create venv"
fi

VENV_PY="${VENV_DIR}/bin/python"
[ -x "${VENV_PY}" ] || fail "${VENV_PY} missing after venv creation"

"${VENV_PY}" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' \
  || fail "venv interpreter is not Python 3.12"

# Guard against a venv that inherits system site-packages.
if "${VENV_PY}" -c 'import sys; print("\n".join(sys.path))' \
    | grep -q '/usr/lib/python3/dist-packages'; then
  fail "venv leaks system dist-packages; recreate with --without-system-site-packages"
fi

log "upgrading pip inside venv"
"${VENV_PY}" -m pip install --upgrade pip \
  || fail "pip upgrade failed"

log "installing project (editable) into venv"
"${VENV_PY}" -m pip install \
  --only-binary=:all: \
  --python-version 3.12 \
  --implementation cp \
  --abi cp312 \
  -e "${REPO_ROOT}/python" \
  || fail "editable install failed"

# Reject any wheel that resolved to a non-cp312 tag.
if "${VENV_PY}" -m pip debug --verbose 2>/dev/null \
    | grep -q 'cp314'; then
  fail "pip compatible tags include cp314; environment is not isolated"
fi

"${VENV_PY}" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' \
  || fail "post-install interpreter check failed"

log "environment ready: ${VENV_PY}"
