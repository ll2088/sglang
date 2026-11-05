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
# Single implementation of the extensible KV cache CI environment setup.
#
# Every other entry point (repo root wrapper, scripts/ wrapper) must source
# this file instead of duplicating the logic, so the interpreter pinning and
# the pip invocation cannot drift apart again.
#
# Usage:
#   source scripts/ci/setup_extensible_kv_cache_env.sh
#   setup_extensible_kv_cache_env
#
# The function is idempotent: calling it twice reuses the existing .venv.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
PYTHON_BIN="python3.12"

log() {
    printf '[setup_extensible_kv_cache_env] %s\n' "$*" >&2
}

fail() {
    log "ERROR: $*"
    log "which ${PYTHON_BIN}: $(command -v "${PYTHON_BIN}" || echo '<not found>')"
    log "${PYTHON_BIN} -V: $(${PYTHON_BIN} -V 2>&1 || echo '<failed>')"
    log "PATH: ${PATH}"
    exit 1
}

assert_venv_python() {
    local py="${VENV_DIR}/bin/python"
    if [[ ! -x "${py}" ]]; then
        fail "${py} is missing or not executable"
    fi
    "${py}" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' \
        || fail "${py} is not Python 3.12"
    "${py}" -m pip --version >/dev/null 2>&1 \
        || fail "${py} -m pip is not usable"
}

print_venv_info() {
    local py="${VENV_DIR}/bin/python"
    log "${py} -V: $(${py} -V 2>&1)"
    "${py}" - <<'PY' >&2
import sys
print(f"sys.executable={sys.executable}")
print(f"sys.prefix={sys.prefix}")
print(f"sys.base_prefix={sys.base_prefix}")
PY
    log "pip path: $(${py} -m pip --version 2>&1)"
}

setup_extensible_kv_cache_env() {
    cd "${REPO_ROOT}"

    if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
        fail "${PYTHON_BIN} is not on PATH"
    fi

    if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
        log "creating venv at ${VENV_DIR} with ${PYTHON_BIN}"
        "${PYTHON_BIN}" -m venv "${VENV_DIR}" \
            || fail "failed to create venv with ${PYTHON_BIN}"
    else
        log "reusing existing venv at ${VENV_DIR}"
    fi

    # Assert the interpreter before installing anything: a wrong interpreter
    # here is what previously pulled cp314 wheels into a 3.12 environment.
    assert_venv_python
    print_venv_info

    # Never let pip fall back to the system interpreter.
    export PIP_REQUIRE_VIRTUALENV=1
    export VIRTUAL_ENV="${VENV_DIR}"
    export PATH="${VENV_DIR}/bin:${PATH}"

    local py="${VENV_DIR}/bin/python"

    "${py}" -m pip install --upgrade pip \
        || fail "failed to upgrade pip inside ${VENV_DIR}"

    # Install the project in editable mode through the venv interpreter only.
    "${py}" -m pip install -e python \
        || fail "failed to install python package into ${VENV_DIR}"

    # Re-assert after installation: a stray system dist-packages entry would
    # show up as a different interpreter or prefix.
    assert_venv_python
    print_venv_info

    log "environment ready"
}
