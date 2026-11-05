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
# The CI image ships a system Python 3.14; installing the project with it
# pulls cp314 wheels and fails. This script creates and activates a 3.12
# venv, installs the project with that interpreter only, and verifies the
# interpreter and pip tags before and after installation.
#
# Usage (from the repository root):
#   source scripts/ci/setup_extensible_kv_cache_env.sh
# or:
#   bash scripts/ci/setup_extensible_kv_cache_env.sh
#
# The script is safe to source: it exports VIRTUAL_ENV and prepends the venv
# bin directory to PATH so later commands in the same shell use it.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
PYTHON_BIN="${PYTHON312_BIN:-python3.12}"

log() {
    printf '[setup_extensible_kv_cache_env] %s\n' "$*" >&2
}

fail() {
    log "ERROR: $*"
    log "which python: $(command -v python || true)"
    log "python -V: $(python -V 2>&1 || true)"
    log "which python3.12: $(command -v python3.12 || true)"
    log "pip -V: $(python -m pip -V 2>&1 || true)"
    exit 1
}

assert_python312() {
    local py="$1"
    "${py}" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' \
        || fail "${py} is not Python 3.12"
}

assert_no_cp314_tags() {
    local py="$1"
    local tags
    tags="$("${py}" -m pip debug --verbose 2>/dev/null | grep -E '^  cp[0-9]+' || true)"
    if printf '%s\n' "${tags}" | grep -q 'cp314'; then
        fail "pip compatible tags still contain cp314; interpreter is not isolated"
    fi
}

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    fail "${PYTHON_BIN} not found; install Python 3.12 or set PYTHON312_BIN"
fi

assert_python312 "${PYTHON_BIN}"

if [ ! -x "${VENV_DIR}/bin/python" ]; then
    log "creating venv at ${VENV_DIR} with ${PYTHON_BIN}"
    "${PYTHON_BIN}" -m venv "${VENV_DIR}" || fail "failed to create venv"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
export VIRTUAL_ENV="${VENV_DIR}"
export PATH="${VENV_DIR}/bin:${PATH}"

assert_python312 "${VENV_DIR}/bin/python"

log "upgrading pip/setuptools/wheel in venv"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel \
    || fail "failed to upgrade pip tooling"

assert_no_cp314_tags "${VENV_DIR}/bin/python"

log "installing project in editable mode with venv python"
"${VENV_DIR}/bin/python" -m pip install -e "${REPO_ROOT}/python" \
    || fail "failed to install project"

assert_python312 "${VENV_DIR}/bin/python"
assert_no_cp314_tags "${VENV_DIR}/bin/python"

log "environment ready: $(command -v python) ($(python -V 2>&1))"
