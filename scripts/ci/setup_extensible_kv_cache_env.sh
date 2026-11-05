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
# This script is the single implementation used by every CI dimension. It
# deliberately avoids the system pip: all installs go through
# ``.venv/bin/python -m pip`` so that PIP_REQUIRE_VIRTUALENV and the interpreter
# assertions actually apply. Any failure exits non-zero before pytest runs.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

log() { printf '[setup_extensible_kv_cache_env] %s\n' "$*" >&2; }

fail() {
  log "ERROR: $*"
  exit 1
}

# 1) Locate python3.12 explicitly and fail loudly when it is missing.
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  log "which python3.12: $(command -v python3.12 || echo '<not found>')"
  log "python3.12 -V: $(python3.12 -V 2>&1 || echo '<not runnable>')"
  fail "$PYTHON_BIN not found; install Python 3.12 before running this script"
fi
log "using interpreter: $(command -v "$PYTHON_BIN") ($("$PYTHON_BIN" -V 2>&1))"

# 2) Create the venv if needed, then assert the interpreter inside it.
if [ ! -x "$VENV_DIR/bin/python" ]; then
  log "creating virtualenv at $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR" || fail "failed to create virtualenv"
fi

VENV_PY="$VENV_DIR/bin/python"
"$VENV_PY" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' \
  || fail "virtualenv interpreter is not Python 3.12"
"$VENV_PY" -c 'import sys; assert sys.prefix != sys.base_prefix, sys.prefix' \
  || fail "virtualenv interpreter is not isolated (sys.prefix == sys.base_prefix)"
log "virtualenv python: $("$VENV_PY" -V 2>&1) prefix=$("$VENV_PY" -c 'import sys; print(sys.prefix)')"

# 3) Make sure no ambient pip configuration redirects installs elsewhere.
unset PIP_TARGET PIP_PREFIX PIP_ROOT PIP_USER PIP_CONFIG_FILE || true
export PIP_REQUIRE_VIRTUALENV=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export VIRTUAL_ENV="$VENV_DIR"
export PATH="$VENV_DIR/bin:$PATH"

log "command -v python: $(command -v python)"
log "python -V: $(python -V 2>&1)"

# 4) Install through the venv interpreter only.
"$VENV_PY" -m pip install --upgrade pip \
  || fail "failed to upgrade pip inside the virtualenv"
"$VENV_PY" -m pip install -e "$REPO_ROOT/python" \
  || fail "failed to install sglang into the virtualenv"

# 5) Re-assert isolation after install, then smoke-test the package.
"$VENV_PY" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' \
  || fail "interpreter changed after install"
"$VENV_PY" -c 'import sys; assert sys.prefix != sys.base_prefix, sys.prefix' \
  || fail "virtualenv isolation lost after install"
"$VENV_PY" -m pip check || fail "pip check reported broken dependencies"
"$VENV_PY" -c 'import sglang' || fail "import sglang failed after install"

log "environment ready: $VENV_DIR"
