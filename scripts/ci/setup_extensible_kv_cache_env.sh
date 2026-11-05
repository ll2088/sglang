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
# Single entry point for preparing the Python 3.12 environment used by the
# extensible KV cache verification jobs. CI must call this script (and only
# this script) so that the interpreter and pip always point at the venv.
set -euxo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

LOG_DIR="${REPO_ROOT}/.ci-logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/setup_extensible_kv_cache_env.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "error: ${PYTHON_BIN} not found on PATH" >&2
  exit 1
fi

# Recreate the venv from scratch so a stale interpreter cannot leak in.
rm -rf "${VENV_DIR}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# Fail loudly if the venv did not take over the shell.
which python
python -V
pip -V
python -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'

# Keep pip from resolving wheels for a different interpreter (e.g. cp314).
export PIP_ONLY_BINARY="${PIP_ONLY_BINARY:-:all:}"
export PIP_REQUIRE_VIRTUALENV="true"
export PIP_DISABLE_PIP_VERSION_CHECK="1"

python -m pip install --upgrade pip
python -m pip install -e python

# Smoke checks before any pytest job runs.
python -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
python -c 'import sglang; print("sglang import ok")'
python -c 'from sglang.srt.mem_cache.vmm_allocator import vmm_unavailable_reason; print("vmm probe:", vmm_unavailable_reason())'

python -m pip list
