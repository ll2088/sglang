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
# Prepare a python3.12 virtualenv for the extensible KV cache tests.
#
# The CI image ships several interpreters and a system site-packages that may
# contain cp314 wheels. Everything below is pinned to python3.12 and to the
# venv's own pip so that dependency resolution never falls back to the system
# interpreter.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

log() { printf '[setup_extensible_kv_cache_env] %s\n' "$*" >&2; }

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  log "${PYTHON_BIN} not found on PATH"
  exit 1
fi

# Rebuild the venv from scratch so a stale or system-linked environment can
# never leak into the test run.
rm -rf "${VENV_DIR}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"

VENV_PYTHON="${VENV_DIR}/bin/python"
VENV_PIP="${VENV_PYTHON} -m pip"

log "which python: $(command -v "${PYTHON_BIN}")"
log "python -V: $("${PYTHON_BIN}" -V 2>&1)"
log "venv python -V: $("${VENV_PYTHON}" -V 2>&1)"
log "venv pip -V: $(${VENV_PIP} -V 2>&1)"

# Fail fast if the venv is not bound to python3.12.
"${VENV_PYTHON}" - <<'PY'
import sys

if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"expected python3.12, got {sys.version.split()[0]}")
PY

# Refuse to install wheels built for another interpreter (e.g. cp314).
# --only-binary=:all: keeps pip from falling back to sdists, and the explicit
# --python-version/--implementation/--abi triple pins resolution to cp312.
PIP_PIN_ARGS=(
  --only-binary=:all:
  --implementation cp
  --python-version 3.12
  --abi cp312
)

${VENV_PIP} install --upgrade pip

# Install the project itself; the pin args above apply to every dependency.
if ! ${VENV_PIP} install "${PIP_PIN_ARGS[@]}" -e "${REPO_ROOT}/python"; then
  log "pip install failed; last 200 lines of pip output follow"
  ${VENV_PIP} install "${PIP_PIN_ARGS[@]}" -e "${REPO_ROOT}/python" 2>&1 | tail -n 200 >&2 || true
  exit 1
fi

# Verify no cp314 wheel slipped in.
if ${VENV_PIP} list --format=freeze | grep -E 'cp314' >/dev/null 2>&1; then
  log "cp314 wheel detected in venv"
  exit 1
fi

log "environment ready: ${VENV_DIR}"
