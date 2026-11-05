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
# Prepare a Python 3.12 virtualenv for the extensible KV cache checks.
#
# The CI runner's system Python may be newer (e.g. 3.14), which makes pip
# resolve cp314 wheels for packages that only publish cp312 wheels. This
# script pins the interpreter, asserts the version, and installs the project
# into a local .venv so every later command uses the same interpreter.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3.12}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "error: ${PYTHON_BIN} not found on PATH" >&2
  echo "available pythons:" >&2
  command -v python3 || true
  python3 -V || true
  exit 1
fi

# Assert the interpreter before creating the venv so a wrong PYTHON_BIN fails
# fast with a clear message instead of resolving cp314 wheels later.
"${PYTHON_BIN}" - <<'PY'
import sys
if sys.version_info[:2] != (3, 12):
    raise SystemExit(
        f"expected Python 3.12, got {sys.version.split()[0]} at {sys.executable}"
    )
print(f"using {sys.executable} ({sys.version.split()[0]})")
PY

if [ ! -d .venv ]; then
  "${PYTHON_BIN}" -m venv .venv
fi

# shellcheck disable=SC1091
. .venv/bin/activate

python - <<'PY'
import sys
if sys.version_info[:2] != (3, 12):
    raise SystemExit(
        f"venv interpreter is {sys.version.split()[0]} at {sys.executable}"
    )
print(f"venv python: {sys.executable} ({sys.version.split()[0]})")
PY

python -m pip install --upgrade pip
# Only resolve wheels built for this interpreter; a missing cp312 wheel is a
# hard error rather than a silent source build against the wrong ABI.
python -m pip install --only-binary=:all: -e python

python -V
python -m pip -V
which python
