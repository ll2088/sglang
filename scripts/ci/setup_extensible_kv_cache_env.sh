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
# The CI runner may have a system Python (3.14) on PATH and PYTHONPATH pointing
# at /usr/lib/python3/dist-packages. Both are cleared here so that every pip
# install and every test run uses the venv interpreter explicitly.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3.12}"

log() { printf '[setup_extensible_kv_cache_env] %s\n' "$*" >&2; }

fail() {
  log "ERROR: $*"
  exit 1
}

# 1. Drop inherited interpreter configuration that would leak the system Python.
unset PYTHONPATH || true
unset PYTHONHOME || true
unset PYTHONSTARTUP || true

# 2. Locate a real Python 3.12 interpreter by absolute path.
if [[ ! -x "${PYTHON_BIN}" ]]; then
  if command -v python3.12 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3.12)"
  else
    fail "python3.12 not found; set PYTHON_BIN to an absolute interpreter path"
  fi
fi
log "using interpreter: ${PYTHON_BIN}"
"${PYTHON_BIN}" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' \
  || fail "interpreter is not Python 3.12"

# 3. Create (or reuse) the venv and always address it by absolute path.
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  log "creating venv at ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}" || fail "failed to create venv"
fi
VENV_PY="${VENV_DIR}/bin/python"
"${VENV_PY}" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' \
  || fail "venv interpreter is not Python 3.12"
log "venv interpreter: $("${VENV_PY}" -c 'import sys; print(sys.executable, sys.version.split()[0])')"

# 4. Install with the venv interpreter only; never bare `pip`.
"${VENV_PY}" -m pip install --upgrade pip setuptools wheel \
  || fail "failed to upgrade pip in venv"
"${VENV_PY}" -m pip install -e "${REPO_ROOT}/python" \
  || fail "failed to install sglang into venv"

# 5. Smoke check: the venv must import sglang and torch, and must not resolve
#    to the system dist-packages.
"${VENV_PY}" - <<'PY' || exit 1
import sys
assert sys.version_info[:2] == (3, 12), sys.version
import sglang  # noqa: F401
import torch  # noqa: F401
print("smoke ok:", sys.executable, sys.version.split()[0])
print("sglang:", getattr(sglang, "__file__", "<unknown>"))
print("torch:", torch.__version__)
PY

log "environment ready: ${VENV_PY}"
