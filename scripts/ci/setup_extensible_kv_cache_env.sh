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
# Prepare an isolated Python 3.12 virtualenv for the extensible KV cache tests.
#
# The CI image defaults to a newer system Python (3.14), so every pip call must
# go through the venv interpreter explicitly. This script never relies on a
# bare `pip` and never reads the system site-packages.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
LOG_FILE="${REPO_ROOT}/.setup_extensible_kv_cache_env.log"

log() {
    echo "[setup_extensible_kv_cache_env] $*"
}

on_error() {
    local exit_code=$?
    echo "[setup_extensible_kv_cache_env] FAILED with exit code ${exit_code}" >&2
    if [[ -f "${LOG_FILE}" ]]; then
        echo "[setup_extensible_kv_cache_env] last 200 log lines:" >&2
        tail -n 200 "${LOG_FILE}" >&2 || true
    fi
    exit "${exit_code}"
}
trap on_error ERR

# Redirect the rest of the output to the log while still showing it.
exec > >(tee -a "${LOG_FILE}") 2>&1

log "repo root: ${REPO_ROOT}"
log "venv dir: ${VENV_DIR}"

# Locate a Python 3.12 interpreter. Prefer the versioned binary so the system
# default (3.14) is never picked up by accident.
PYTHON312=""
for candidate in python3.12 /usr/bin/python3.12 /usr/local/bin/python3.12; do
    if command -v "${candidate}" >/dev/null 2>&1; then
        PYTHON312="$(command -v "${candidate}")"
        break
    fi
done
if [[ -z "${PYTHON312}" ]]; then
    log "python3.12 not found; available interpreters:"
    ls -1 /usr/bin/python3* 2>/dev/null || true
    log "install python3.12 before running this script"
    exit 1
fi
log "using interpreter: ${PYTHON312}"
"${PYTHON312}" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'

# Recreate the venv from scratch so no stale or system packages leak in.
rm -rf "${VENV_DIR}"
"${PYTHON312}" -m venv --clear "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

log "which python: $(command -v python)"
log "python -V: $(python -V 2>&1)"
log "which pip: $(command -v pip || true)"
log "pip -V: $(python -m pip -V 2>&1)"
log "VIRTUAL_ENV: ${VIRTUAL_ENV:-<unset>}"

# Hard assertions: the interpreter must be 3.12 and live inside the venv.
python -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
python - <<'PY'
import os
import sys

venv = os.environ.get("VIRTUAL_ENV")
assert venv, "VIRTUAL_ENV is not set; venv activation failed"
exe = os.path.realpath(sys.executable)
assert exe.startswith(os.path.realpath(venv) + os.sep), (
    f"interpreter {exe} is outside venv {venv}"
)
print(f"interpreter ok: {exe}")
PY

# Never read the system site-packages.
export PIP_NO_SYSTEM_SITE_PACKAGES=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

python -m pip install --upgrade pip setuptools wheel

# Install the project in editable mode through the venv interpreter only.
# --ignore-installed avoids "Requirement already satisfied" from the system
# dist-packages that the CI image exposes.
python -m pip install --ignore-installed -e "${REPO_ROOT}/python"

# Verify the installed package resolves inside the venv.
python - <<'PY'
import os
import sys

import sglang

venv = os.path.realpath(os.environ["VIRTUAL_ENV"])
path = os.path.realpath(sglang.__file__)
assert path.startswith(venv + os.sep), (
    f"sglang resolved to {path}, outside venv {venv}"
)
print(f"sglang ok: {path}")
PY

log "environment ready"
