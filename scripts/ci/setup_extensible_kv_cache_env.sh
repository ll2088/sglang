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
# The CI environment must not rely on whatever `python`/`pip` happen to be on
# PATH: a stray 3.14 interpreter would make pip resolve cp314 wheels and the
# 3.12 assertions in the test commands would never run. This script creates
# the venv explicitly with python3.12, activates it, and installs everything
# through `python -m pip` so the interpreter is unambiguous.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"

log() { printf '[setup_extensible_kv_cache_env] %s\n' "$*" >&2; }

# 1) Require python3.12 before touching anything else.
if ! command -v python3.12 >/dev/null 2>&1; then
  log "ERROR: python3.12 not found on PATH."
  log "Install Python 3.12 (e.g. apt-get install python3.12 python3.12-venv)"
  log "or point PATH at a 3.12 interpreter before running this script."
  exit 1
fi
log "python3.12: $(command -v python3.12)"

# 2) Create the venv with python3.12 and activate it.
if [ ! -x "${VENV_DIR}/bin/python" ]; then
  log "Creating venv at ${VENV_DIR}"
  python3.12 -m venv "${VENV_DIR}"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# 3) Diagnostics so a wrong interpreter is obvious in CI logs.
log "python -V: $(python -V 2>&1)"
log "sys.executable: $(python -c 'import sys; print(sys.executable)')"
log "sysconfig.get_platform(): $(python -c 'import sysconfig; print(sysconfig.get_platform())')"
log "python -m pip -V: $(python -m pip -V 2>&1)"

# 4) Hard assertions: interpreter is 3.12 and lives inside the venv.
python - <<'PY'
import os
import sys

assert sys.version_info[:2] == (3, 12), f"expected Python 3.12, got {sys.version}"
venv = os.environ.get("VIRTUAL_ENV")
assert venv, "VIRTUAL_ENV is not set; venv activation failed"
assert os.path.realpath(sys.executable).startswith(os.path.realpath(venv) + os.sep), (
    f"sys.executable {sys.executable} is not inside VIRTUAL_ENV {venv}"
)
PY

# 5) Install through `python -m pip` only, so PATH pip cannot leak in.
python -m pip install --upgrade pip
python -m pip install -e "${REPO_ROOT}/python"

# 6) Smoke checks: import and test collection must work before CI runs tests.
python -c 'import sglang'
python -m pytest --collect-only -q \
  "${REPO_ROOT}/tests/srt/mem_cache" \
  "${REPO_ROOT}/tests/srt/arg_groups"

log "Environment ready: $(python -V 2>&1) at $(python -c 'import sys; print(sys.executable)')"
