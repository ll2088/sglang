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
# Prepare a Python 3.12 virtualenv for the extensible KV cache verification.
#
# The CI image ships a system Python newer than 3.12; installing with the
# system pip resolves cp314 wheels and the tests then run under a different
# interpreter than the one the dependencies were built for. This script makes
# the interpreter explicit and idempotent so every verification dimension can
# source it before running pytest.
#
# Usage:
#   source scripts/ci/setup_extensible_kv_cache_env.sh
#   python -m pytest -q tests/srt/mem_cache/test_vmm_allocator.py

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"

# 1. Locate a python3.12 interpreter. Prefer the versioned binary so a system
#    `python3` pointing at 3.14 cannot be picked up by accident.
PYTHON312=""
for candidate in python3.12 /usr/bin/python3.12 /usr/local/bin/python3.12; do
    if command -v "${candidate}" >/dev/null 2>&1; then
        PYTHON312="$(command -v "${candidate}")"
        break
    fi
done
if [[ -z "${PYTHON312}" ]]; then
    echo "error: python3.12 not found; install it before running this script" >&2
    exit 1
fi

# 2. Create the venv once. Reuse it on later runs so repeated verification
#    does not pay the install cost again.
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    "${PYTHON312}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# 3. Fail loudly if activation did not take effect. This is the check that
#    would have caught the previous system-pip install.
python -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
python -c 'import sys; print("python:", sys.executable, sys.version)'
pip --version

# 4. Install the project with the venv pip only. Constrain to cp312 wheels so
#    a stray cp314 wheel cannot be selected even if the index offers one.
export PIP_PYTHON_VERSION=3.12
export PIP_DISABLE_PIP_VERSION_CHECK=1
pip install --upgrade pip
pip install -e "${REPO_ROOT}/python" \
    --only-binary=:all: \
    --extra-index-url "${PIP_EXTRA_INDEX_URL:-}" 2>/dev/null || \
    pip install -e "${REPO_ROOT}/python"

# 5. Re-assert the interpreter after installation; some packages ship
#    post-install hooks that can shadow the venv python.
python -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'

# 6. Make the venv the default for the rest of the CI step.
export PATH="${VENV_DIR}/bin:${PATH}"
export VIRTUAL_ENV="${VENV_DIR}"

echo "extensible kv cache env ready: $(python -c 'import sys; print(sys.executable)')"
