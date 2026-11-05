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
# Prepare a python3.12 virtualenv for the extensible KV cache verification.
#
# The verification commands assume `.venv/bin/activate` exists and that the
# interpreter inside it is python3.12. This script creates that venv with an
# explicit python3.12 binary (never the system default `python3`), installs the
# project in editable mode into it, and asserts the interpreter and pip are
# bound to the venv before returning.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"

log() { printf '[setup-extensible-kv-cache] %s\n' "$*" >&2; }

fail() {
  log "ERROR: $*"
  log "--- diagnostics ---"
  command -v python3.12 >&2 || true
  python3.12 -V >&2 || true
  command -v python3 >&2 || true
  python3 -V >&2 || true
  if [ -x "${VENV_DIR}/bin/python" ]; then
    "${VENV_DIR}/bin/python" -V >&2 || true
    "${VENV_DIR}/bin/python" -m pip -V >&2 || true
    "${VENV_DIR}/bin/python" -c 'import sys; print(sys.executable); print(sys.path)' >&2 || true
  fi
  "${VENV_DIR}/bin/python" -m pip config list >&2 || true
  exit 1
}

if ! command -v python3.12 >/dev/null 2>&1; then
  fail "python3.12 not found on PATH"
fi

log "python3.12: $(command -v python3.12) ($(python3.12 -V 2>&1))"

if [ ! -x "${VENV_DIR}/bin/python" ]; then
  log "creating venv at ${VENV_DIR}"
  python3.12 -m venv "${VENV_DIR}"
fi

VENV_PY="${VENV_DIR}/bin/python"

# Never let pip fall back to the system interpreter or global site-packages.
export PIP_REQUIRE_VIRTUALENV=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

log "venv python: $("${VENV_PY}" -V 2>&1)"
"${VENV_PY}" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' \
  || fail "venv interpreter is not python3.12"

PIP_INFO="$("${VENV_PY}" -m pip -V)"
log "venv pip: ${PIP_INFO}"
case "${PIP_INFO}" in
  *"${VENV_DIR}"*) ;;
  *) fail "pip is not bound to ${VENV_DIR}" ;;
esac

log "installing project (editable) into venv"
"${VENV_PY}" -m pip install --no-cache-dir -e "${REPO_ROOT}/python"

log "environment ready"
