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
# Single entry point for preparing the extensible KV cache check environment.
#
# The checks must run under Python 3.12. This script creates (or reuses) a
# local .venv with python3.12, installs the project with the venv's own pip,
# and refuses to continue if the interpreter or pip does not point into the
# venv. It never invokes a bare `pip`.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
PYTHON_BIN="${VENV_DIR}/bin/python"
PIP_BIN="${VENV_DIR}/bin/pip"

log() {
    printf '[setup_extensible_kv_cache_env] %s\n' "$*" >&2
}

fail() {
    log "ERROR: $*"
    exit 1
}

# 1. Locate a python3.12 interpreter before touching any venv.
if ! command -v python3.12 >/dev/null 2>&1; then
    fail "python3.12 not found on PATH; install it before running this script"
fi

# 2. Create the venv if it is missing or was built with another interpreter.
if [ ! -x "${PYTHON_BIN}" ]; then
    log "creating venv at ${VENV_DIR} with python3.12"
    python3.12 -m venv "${VENV_DIR}"
fi

# 3. Assert the venv interpreter is exactly 3.12 and lives inside the venv.
"${PYTHON_BIN}" - <<'PY' || exit 1
import sys

if sys.version_info[:2] != (3, 12):
    raise SystemExit(
        f"expected Python 3.12, got {sys.version.split()[0]} at {sys.executable}"
    )
if ".venv" not in sys.executable:
    raise SystemExit(f"interpreter is not inside .venv: {sys.executable}")
print(f"interpreter ok: {sys.executable} ({sys.version.split()[0]})")
PY

# 4. Assert pip resolves to the venv interpreter, not a system pip.
"${PYTHON_BIN}" -m pip --version >/dev/null 2>&1 \
    || fail "venv pip is not usable; recreate ${VENV_DIR}"
PIP_EXECUTABLE="$("${PYTHON_BIN}" -c 'import sys; print(sys.executable)')"
case "${PIP_EXECUTABLE}" in
    "${VENV_DIR}"/*) ;;
    *) fail "pip would run under ${PIP_EXECUTABLE}, expected ${VENV_DIR}" ;;
esac

# 5. Print diagnostics so CI logs show which interpreter is in use.
log "which python: $(command -v python || true)"
log "python -V: $(python -V 2>&1 || true)"
log "venv python: ${PYTHON_BIN}"
log "venv python -V: $("${PYTHON_BIN}" -V 2>&1)"
log "venv pip: $("${PYTHON_BIN}" -m pip -V)"

# 6. Install with the venv's pip only. PIP_* and PYTHONPATH from the CI image
#    must not redirect resolution to another interpreter's wheels.
unset PYTHONPATH || true
unset PIP_TARGET PIP_PREFIX PIP_ROOT || true

log "installing project into ${VENV_DIR}"
"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install -e "${REPO_ROOT}/python"

# 7. Smoke checks: the package imports and pytest is available.
"${PYTHON_BIN}" -c 'import sglang' \
    || fail "smoke check failed: import sglang"
"${PYTHON_BIN}" -m pytest --version \
    || fail "smoke check failed: pytest --version"

log "environment ready: ${VENV_DIR}"
