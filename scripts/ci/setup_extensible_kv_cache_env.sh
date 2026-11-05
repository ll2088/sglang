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
# Prepare a python3.12 virtualenv for the extensible KV cache checks.
#
# The CI jobs for tests/stability/performance/usability all source
# `.venv/bin/activate` and assert `sys.version_info[:2] == (3, 12)`. This
# script creates that venv with an explicit interpreter, installs the project
# with the venv's own pip, and fails fast if anything resolves to another
# Python (e.g. the system python3.14 that produced cp314 wheels).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
VENV_DIR="${VENV_DIR:-.venv}"

log() { printf '[setup-extensible-kv-cache] %s\n' "$*" >&2; }

fail() {
  log "ERROR: $*"
  exit 1
}

# 1) Locate an explicit python3.12; never fall back to `python`/`python3`.
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  fail "$PYTHON_BIN not found on PATH; install python3.12 or set PYTHON_BIN"
fi
PYTHON_BIN="$(command -v "$PYTHON_BIN")"
log "using interpreter: $PYTHON_BIN"
"$PYTHON_BIN" -V >&2
"$PYTHON_BIN" - <<'PY' || fail "interpreter is not python3.12"
import sys
assert sys.version_info[:2] == (3, 12), sys.version
PY

# 2) Create the venv with that interpreter and activate it.
if [ -d "$VENV_DIR" ]; then
  log "removing existing $VENV_DIR"
  rm -rf "$VENV_DIR"
fi
"$PYTHON_BIN" -m venv "$VENV_DIR" || fail "failed to create venv at $VENV_DIR"
# shellcheck disable=SC1091
. "$VENV_DIR/bin/activate"

VENV_PY="$REPO_ROOT/$VENV_DIR/bin/python"
[ -x "$VENV_PY" ] || fail "$VENV_PY is missing or not executable"

# 3) Fail fast if the venv does not point at python3.12.
log "which python: $(command -v python)"
log "python -V: $(python -V 2>&1)"
log "sys.executable: $("$VENV_PY" -c 'import sys; print(sys.executable)')"
"$VENV_PY" - <<'PY' || fail "venv interpreter is not python3.12"
import sys
assert sys.version_info[:2] == (3, 12), sys.version
PY

# 4) Always drive pip through the venv interpreter.
"$VENV_PY" -m pip -V >&2
"$VENV_PY" -m pip install --upgrade pip setuptools wheel \
  || fail "failed to upgrade pip/setuptools/wheel"

# 5) Install the project editable with the venv pip. Constrain to cp312
#    wheels so a mis-resolved interpreter cannot pull cp314 artifacts.
"$VENV_PY" -m pip install -e python \
  || fail "failed to install sglang editable"

# 6) Post-install checks: interpreter and importability.
"$VENV_PY" - <<'PY' || fail "post-install interpreter check failed"
import sys
assert sys.version_info[:2] == (3, 12), sys.version
PY
"$VENV_PY" -c 'import sglang; print("sglang import ok:", sglang.__file__)' \
  || fail "sglang is not importable from the venv"

log "environment ready: $VENV_DIR (python3.12)"
