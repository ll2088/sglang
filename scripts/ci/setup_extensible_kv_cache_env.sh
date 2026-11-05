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
# Idempotent environment bootstrap for the extensible KV cache checks.
#
# Creates (or reuses) a Python 3.12 virtualenv at .venv, verifies that both
# `python` and `pip` resolve inside it, and installs the project with wheels
# restricted to cp312 so a system Python 3.14 pip can never leak in.
#
# Usage:  bash scripts/ci/setup_extensible_kv_cache_env.sh
#         source .venv/bin/activate

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

log() { printf '[setup-extensible-kv-cache] %s\n' "$*" >&2; }

fail() {
  log "ERROR: $*"
  log "python: $(command -v python || echo '<none>')"
  log "pip:    $(command -v pip || echo '<none>')"
  exit 1
}

# 1. Locate a real Python 3.12 interpreter before touching the venv.
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  fail "${PYTHON_BIN} not found; install Python 3.12 or set PYTHON_BIN"
fi
PY_VERSION="$("${PYTHON_BIN}" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
if [ "${PY_VERSION}" != "3.12" ]; then
  fail "${PYTHON_BIN} reports Python ${PY_VERSION}, expected 3.12"
fi

# 2. Create the venv, or validate an existing one. Never reuse a broken venv.
if [ -d "${VENV_DIR}" ]; then
  if [ ! -x "${VENV_DIR}/bin/python" ]; then
    log "removing incomplete venv at ${VENV_DIR}"
    rm -rf "${VENV_DIR}"
  else
    EXISTING_VERSION="$("${VENV_DIR}/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo unknown)"
    if [ "${EXISTING_VERSION}" != "3.12" ]; then
      log "existing venv is Python ${EXISTING_VERSION}; recreating"
      rm -rf "${VENV_DIR}"
    fi
  fi
fi

if [ ! -d "${VENV_DIR}" ]; then
  log "creating venv at ${VENV_DIR} with ${PYTHON_BIN}"
  if ! "${PYTHON_BIN}" -m venv "${VENV_DIR}"; then
    rm -rf "${VENV_DIR}"
    fail "failed to create venv at ${VENV_DIR}"
  fi
fi

# 3. Activate and hard-verify interpreter and pip paths.
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

PY_EXEC="$(python -c 'import sys; print(sys.executable)')"
case "${PY_EXEC}" in
  "${VENV_DIR}"/*) ;;
  *) fail "python resolves to ${PY_EXEC}, not inside ${VENV_DIR}" ;;
esac

PY_VER="$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
if [ "${PY_VER}" != "3.12" ]; then
  fail "active python is ${PY_VER}, expected 3.12 (${PY_EXEC})"
fi

PIP_EXEC="$(python -m pip --version | awk '{print $NF}')"
case "${PIP_EXEC}" in
  "${VENV_DIR}"/*) ;;
  *) fail "pip resolves to ${PIP_EXEC}, not inside ${VENV_DIR}" ;;
esac

log "python: ${PY_EXEC} (${PY_VER})"
log "pip:    ${PIP_EXEC}"

# 4. Upgrade pip and install with cp312-only wheels.
export PIP_ONLY_BINARY=":all:"
export PIP_DISABLE_PIP_VERSION_CHECK=1

python -m pip install --upgrade pip

if ! python -m pip install -e "${REPO_ROOT}/python"; then
  log "pip install failed; dumping diagnostics"
  python -m pip debug -v || true
  python -c 'import sys; print(sys.executable)' || true
  exit 1
fi

log "environment ready; run: source .venv/bin/activate"
