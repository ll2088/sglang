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
# The runner's default interpreter may be newer than the versions the project
# publishes wheels for, which makes pip resolve cp314 wheels and fail. This
# script always creates the venv from python3.12 and verifies the interpreter
# before and after installing dependencies.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "error: ${PYTHON_BIN} not found on PATH" >&2
  exit 1
fi

if [ ! -x "${VENV_DIR}/bin/python" ]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

VENV_PYTHON="${VENV_DIR}/bin/python"

# Never fall back to the system interpreter: fail loudly instead.
if [ ! -x "${VENV_PYTHON}" ]; then
  echo "error: ${VENV_PYTHON} is missing" >&2
  exit 1
fi

echo "which python: $(command -v python || true)"
echo "venv python: ${VENV_PYTHON}"
echo "venv version: $(${VENV_PYTHON} -V 2>&1)"

if ! "${VENV_PYTHON}" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)'; then
  echo "error: ${VENV_PYTHON} is not Python 3.12.x" >&2
  "${VENV_PYTHON}" -V >&2 || true
  exit 1
fi

"${VENV_PYTHON}" -m pip install -U pip
"${VENV_PYTHON}" -m pip install -e "${REPO_ROOT}/python"

echo "installed version: $(${VENV_PYTHON} -V 2>&1)"
if ! "${VENV_PYTHON}" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)'; then
  echo "error: interpreter changed during install" >&2
  exit 1
fi

echo "extensible kv cache env ready: ${VENV_PYTHON}"
