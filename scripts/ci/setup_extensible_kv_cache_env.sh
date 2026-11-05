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
# The CI runner ships a system python (possibly 3.14) whose dist-packages and
# pip configuration leak into dependency resolution. This script isolates the
# install: it clears PIP_*/PYTHONPATH/PYTHONHOME, creates (or reuses) a venv,
# and always invokes pip through "$VENV/bin/python" -m pip --isolated.
set -euxo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="${VENV:-$REPO_ROOT/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

# Drop inherited configuration that could redirect pip or shadow the venv.
unset PIP_INDEX_URL PIP_EXTRA_INDEX_URL PIP_CONFIG_FILE PYTHONPATH PYTHONHOME || true
unset PIP_TARGET PIP_PREFIX PIP_USER || true
export PIP_DISABLE_PIP_VERSION_CHECK=1

if [ ! -x "$VENV/bin/python" ]; then
  "$PYTHON_BIN" -m venv "$VENV"
fi

"$VENV/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'

# Diagnostics so a mis-bound interpreter is obvious in CI logs.
which python || true
python -V || true
python -m pip -V || true
"$VENV/bin/python" -c 'import sys; print("sys.executable=", sys.executable); print("sys.prefix=", sys.prefix)'
"$VENV/bin/python" -m pip -V

"$VENV/bin/python" -m pip --isolated install --no-cache-dir --upgrade pip setuptools wheel
"$VENV/bin/python" -m pip --isolated install --no-cache-dir -e "$REPO_ROOT/python"

# Assert the install did not pull wheels for another interpreter or fall back
# to the system dist-packages.
"$VENV/bin/python" -m pip freeze > "$VENV/pip-freeze.txt"
if grep -E 'cp3(13|14)' "$VENV/pip-freeze.txt"; then
  echo "unexpected non-3.12 wheel in pip freeze" >&2
  exit 1
fi
"$VENV/bin/python" - <<'PY'
import sys

bad = [p for p in sys.path if "/usr/lib/python3/dist-packages" in p]
assert not bad, f"system dist-packages leaked into sys.path: {bad}"
import sglang  # noqa: F401
import pydantic  # noqa: F401
import pybase64  # noqa: F401
print("extensible kv cache env ready")
PY
