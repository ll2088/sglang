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
# The CI image ships a system Python newer than 3.12, so installing with the
# bare ``pip`` resolves wheels for the system interpreter (e.g. cp314) and the
# later ``. .venv/bin/activate`` steps fail. This script always creates the
# venv from an explicit python3.12 binary and installs through
# ``python -m pip`` so the interpreter and pip cannot diverge.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
PYTHON312="${PYTHON312:-/usr/bin/python3.12}"

fail() {
    echo "[setup_extensible_kv_cache_env] ERROR: $*" >&2
    echo "--- diagnostics ---" >&2
    echo "which python: $(command -v python || true)" >&2
    echo "python -V: $(python -V 2>&1 || true)" >&2
    echo "sys.executable: $(python -c 'import sys; print(sys.executable)' 2>/dev/null || true)" >&2
    echo "sys.prefix: $(python -c 'import sys; print(sys.prefix)' 2>/dev/null || true)" >&2
    echo "python -m pip -V: $(python -m pip -V 2>&1 || true)" >&2
    exit 1
}

if [[ ! -x "${PYTHON312}" ]]; then
    fail "${PYTHON312} not found or not executable; set PYTHON312 to a Python 3.12 binary"
fi

"${PYTHON312}" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' \
    || fail "${PYTHON312} is not Python 3.12"

rm -rf "${VENV_DIR}"
"${PYTHON312}" -m venv "${VENV_DIR}" || fail "failed to create venv at ${VENV_DIR}"

# shellcheck disable=SC1091
. "${VENV_DIR}/bin/activate"

python -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' \
    || fail "venv python is not 3.12"
python -c "import sys; assert sys.prefix == '${VENV_DIR}', sys.prefix" \
    || fail "venv python does not report ${VENV_DIR} as sys.prefix"

# Keep pip/setuptools/wheel in lockstep with the venv interpreter before any
# dependency resolution, so no wheel for another CPython ABI is selected.
python -m pip install --upgrade pip setuptools wheel \
    || fail "failed to upgrade pip/setuptools/wheel in venv"
python -m pip -V | grep -q 'cp312' \
    || fail "venv pip is not built for cp312: $(python -m pip -V)"

python -m pip install -e "${REPO_ROOT}/python" \
    || fail "failed to install sglang into venv"

SGLANG_LOCATION="$(python -m pip show sglang | awk '/^Location:/ {print $2}')"
case "${SGLANG_LOCATION}" in
    "${VENV_DIR}"*) ;;
    *) fail "sglang installed outside venv: ${SGLANG_LOCATION}" ;;
esac

echo "[setup_extensible_kv_cache_env] ready: $(python -V), $(python -m pip -V)"
