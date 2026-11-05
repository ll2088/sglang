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
# Usage (must be sourced so the venv stays active for the caller):
#   source scripts/ci/setup_extensible_kv_cache_env.sh
#
# The script is intentionally strict: it fails loudly instead of silently
# falling back to the system interpreter, which previously resolved cp314
# wheels and installed into /usr/lib/python3/dist-packages.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

log() { printf '[setup_extensible_kv_cache_env] %s\n' "$*" >&2; }

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  log "error: ${PYTHON_BIN} not found on PATH"
  exit 1
fi

# Recreate the venv when it is missing or was built by another interpreter.
if [ -d "${VENV_DIR}" ]; then
  if ! "${VENV_DIR}/bin/python" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)' >/dev/null 2>&1; then
    log "existing ${VENV_DIR} is not python3.12; recreating"
    rm -rf "${VENV_DIR}"
  fi
fi

if [ ! -d "${VENV_DIR}" ]; then
  log "creating venv at ${VENV_DIR} with ${PYTHON_BIN}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# Hard guard: never let the system interpreter leak into the install step.
if ! python -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)'; then
  log "error: active python is not 3.12"
  python -V >&2 || true
  exit 1
fi

if ! python -c 'import sys; raise SystemExit(0 if sys.prefix != sys.base_prefix else 1)'; then
  log "error: python is not running inside a virtualenv"
  exit 1
fi

log "which python: $(command -v python)"
log "python -V: $(python -V 2>&1)"
log "pip -V: $(python -m pip -V 2>&1)"
log "sys.prefix: $(python -c 'import sys; print(sys.prefix)')"

# Upgrade pip inside the venv only, then install the project editable.
python -m pip install --upgrade pip
python -m pip install -e "${REPO_ROOT}/python"

log "environment ready"
