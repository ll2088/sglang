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
# Prepare an isolated python3.12 virtualenv for the extensible KV cache tests.
#
# The CI image may default to a newer interpreter (e.g. 3.14), which makes pip
# resolve cp314 wheels that cannot be imported by the 3.12 venv. Every pip
# invocation therefore goes through .venv/bin/python -m pip, and the script
# asserts the interpreter version before and after installation.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Avoid leaking the system interpreter or user site-packages into the venv.
unset PYTHONPATH || true
unset PYTHONHOME || true
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INPUT=1

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "error: $PYTHON_BIN not found; install python3.12 first" >&2
  exit 1
fi

rm -rf .venv
"$PYTHON_BIN" -m venv .venv

VENV_PYTHON=".venv/bin/python"
VENV_PIP=("$VENV_PYTHON" -m pip)

"$VENV_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'

"${VENV_PIP[@]}" install --upgrade pip setuptools wheel

# Install the project in editable mode without pulling optional heavy extras.
if ! "${VENV_PIP[@]}" install -e . ; then
  echo "error: pip install failed" >&2
  echo "--- python -V ---" >&2
  "$VENV_PYTHON" -V >&2 || true
  echo "--- which python ---" >&2
  command -v "$VENV_PYTHON" >&2 || true
  echo "--- pip -V ---" >&2
  "${VENV_PIP[@]}" -V >&2 || true
  exit 1
fi

"$VENV_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
"$VENV_PYTHON" -c 'import sglang; print("sglang import ok")'

echo "extensible kv cache test environment ready: $("$VENV_PYTHON" -V 2>&1)"
