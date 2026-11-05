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
# Prepare a reproducible environment for the extensible KV cache tests.
#
# The previous CI failure happened during dependency resolution because the
# runner picked up Python 3.14 and tried to install cp314 wheels. This script
# pins the interpreter, creates an isolated virtual environment, and installs
# the project into it so that the test commands in the verification plan can
# run against a known Python version.
#
# Usage:
#   bash scripts/ci/setup_extensible_kv_cache_env.sh
#   . .venv/bin/activate
#   python -m pytest -q tests/srt/mem_cache/test_vmm_allocator.py ...

set -euo pipefail

REQUIRED_PYTHON_MAJOR=3
REQUIRED_PYTHON_MINOR=12
VENV_DIR="${VENV_DIR:-.venv}"

log() {
    printf '[setup_extensible_kv_cache_env] %s\n' "$*" >&2
}

fail() {
    log "ERROR: $*"
    exit 1
}

# 1. Locate a Python interpreter matching the required version.
PYTHON_BIN=""
for candidate in "python${REQUIRED_PYTHON_MAJOR}.${REQUIRED_PYTHON_MINOR}" python3 python; do
    if ! command -v "${candidate}" >/dev/null 2>&1; then
        continue
    fi
    version="$("${candidate}" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true)"
    if [ "${version}" = "${REQUIRED_PYTHON_MAJOR}.${REQUIRED_PYTHON_MINOR}" ]; then
        PYTHON_BIN="$(command -v "${candidate}")"
        break
    fi
done

if [ -z "${PYTHON_BIN}" ]; then
    fail "Python ${REQUIRED_PYTHON_MAJOR}.${REQUIRED_PYTHON_MINOR} not found; install it before running this script"
fi

log "Using interpreter: ${PYTHON_BIN}"
"${PYTHON_BIN}" -c 'import sys; print(sys.version)' >&2

# 2. Create (or reuse) an isolated virtual environment.
if [ ! -x "${VENV_DIR}/bin/python" ]; then
    log "Creating virtual environment at ${VENV_DIR}"
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

VENV_PYTHON="${VENV_DIR}/bin/python"
if [ ! -x "${VENV_PYTHON}" ]; then
    fail "virtual environment interpreter ${VENV_PYTHON} is missing"
fi

venv_version="$("${VENV_PYTHON}" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
if [ "${venv_version}" != "${REQUIRED_PYTHON_MAJOR}.${REQUIRED_PYTHON_MINOR}" ]; then
    fail "virtual environment uses Python ${venv_version}, expected ${REQUIRED_PYTHON_MAJOR}.${REQUIRED_PYTHON_MINOR}"
fi

# 3. Upgrade packaging tools inside the venv only.
log "Upgrading pip/setuptools/wheel"
"${VENV_PYTHON}" -m pip install --upgrade pip setuptools wheel

# 4. Install the project in editable mode using the pinned interpreter.
if [ -f "python/pyproject.toml" ] || [ -f "python/setup.py" ]; then
    log "Installing project from python/"
    "${VENV_PYTHON}" -m pip install -e python
else
    log "Installing project from repository root"
    "${VENV_PYTHON}" -m pip install -e .
fi

# 5. Sanity check: the package must import under the pinned interpreter.
log "Verifying sglang import"
"${VENV_PYTHON}" -c 'import sglang; print("sglang import OK")'

log "Environment ready. Activate with: . ${VENV_DIR}/bin/activate"
