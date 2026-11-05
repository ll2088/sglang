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
# Single entry point for preparing the extensible KV cache CI environment.
#
# Every install and check goes through .venv/bin/python so the interpreter is
# pinned to python3.12 and never falls back to the system pip (which may be a
# different minor version and resolve cp314 wheels).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
PYTHON_BIN="${VENV_DIR}/bin/python"

assert_venv_python() {
  "${PYTHON_BIN}" - <<'PY'
import sys
assert sys.version_info[:2] == (3, 12), sys.version
assert sys.prefix.endswith(".venv"), sys.prefix
print("interpreter:", sys.executable)
print("version:", sys.version.split()[0])
print("prefix:", sys.prefix)
PY
}

if ! command -v python3.12 >/dev/null 2>&1; then
  echo "python3.12 is required but was not found on PATH" >&2
  exit 1
fi

if [ ! -x "${PYTHON_BIN}" ]; then
  rm -rf "${VENV_DIR}"
  python3.12 -m venv "${VENV_DIR}"
fi

# Fail fast if the venv is not the interpreter we expect.
assert_venv_python

"${PYTHON_BIN}" -m pip install --upgrade pip
# --require-virtualenv keeps pip from ever installing into the system
# interpreter if the venv activation is lost.
"${PYTHON_BIN}" -m pip install --require-virtualenv -e "${REPO_ROOT}/python"

# Re-assert after installation: a broken venv or a stray system install would
# otherwise only surface much later as a confusing import error.
assert_venv_python

echo "extensible KV cache environment ready: ${PYTHON_BIN}"
