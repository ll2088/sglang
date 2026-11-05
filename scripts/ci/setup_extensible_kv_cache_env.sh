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
# This script is the single source of truth for CI environment preparation of
# the VMM two-phase sizing tests. It is intentionally strict: any failure exits
# non-zero so the CI job fails at the environment step instead of silently
# running with the wrong interpreter.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

LOG_FILE="${REPO_ROOT}/ci_pip.log"
: > "${LOG_FILE}"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[setup] repo root: ${REPO_ROOT}"
echo "[setup] host python: $(command -v python3 || true)"
python3 -V || true

if ! command -v python3.12 >/dev/null 2>&1; then
  echo "[setup] python3.12 not found on PATH" >&2
  exit 1
fi

rm -rf .venv
python3.12 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

python -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
echo "[setup] venv python: $(command -v python)"
python -V

python -m pip install -U pip
python -m pip --version
python -m pip debug --verbose || true

# Constrain resolution to the venv interpreter so no cp314 wheels are pulled.
export PIP_ONLY_BINARY="${PIP_ONLY_BINARY:-:all:}"
export PIP_DISABLE_PIP_VERSION_CHECK=1

python -m pip install --only-binary=:all: -e python

python -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
echo "[setup] environment ready"
