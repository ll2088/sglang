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
# The CI runner may expose several interpreters (including 3.14 wheels from
# the system pip). This script pins every install and every check to the
# .venv created from an absolute python3.12 path, and fails loudly instead of
# silently falling back to the system interpreter.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"

find_python312() {
    local candidate
    for candidate in /usr/bin/python3.12 /usr/local/bin/python3.12 "$(command -v python3.12 || true)"; do
        if [ -n "${candidate}" ] && [ -x "${candidate}" ]; then
            echo "${candidate}"
            return 0
        fi
    done
    return 1
}

PYTHON312="$(find_python312 || true)"
if [ -z "${PYTHON312}" ]; then
    echo "ERROR: python3.12 not found; install python3.12 and python3.12-venv" >&2
    exit 1
fi

if ! "${PYTHON312}" -c 'import venv' >/dev/null 2>&1; then
    echo "ERROR: ${PYTHON312} cannot import venv; install python3.12-venv" >&2
    exit 1
fi

if [ ! -d "${VENV_DIR}" ]; then
    "${PYTHON312}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

assert_venv() {
    local stage="$1"
    echo "[${stage}] which python: $(command -v python)"
    echo "[${stage}] python -V: $(python -V 2>&1)"
    python - <<'PY'
import sys
print(f"sys.executable={sys.executable}")
print(f"sys.prefix={sys.prefix}")
print(f"sys.version_info={sys.version_info[:3]}")
assert sys.version_info[:2] == (3, 12), sys.version_info
assert ".venv" in sys.prefix, sys.prefix
PY
}

assert_venv "before-install"

python -m pip install --upgrade pip
python -m pip install -e "${REPO_ROOT}/python"

assert_venv "after-install"

python -c 'import sglang'
python -m pytest --collect-only -q "${REPO_ROOT}/tests/srt/mem_cache/test_vmm_allocator.py" "${REPO_ROOT}/tests/srt/mem_cache/test_extensible_kv_cache.py" "${REPO_ROOT}/tests/srt/arg_groups/test_arg_utils_extensible_kv_cache.py"

echo "extensible KV cache environment ready: $(command -v python)"
