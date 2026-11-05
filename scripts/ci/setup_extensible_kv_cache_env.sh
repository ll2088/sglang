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
# The CI runner may have a newer system Python (e.g. 3.14) as `python`/`pip`.
# This script pins every step to python3.12 so that dependency resolution does
# not download cp314 wheels into a 3.12 environment.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

if ! command -v python3.12 >/dev/null 2>&1; then
  echo "error: python3.12 not found on PATH" >&2
  echo "       install python3.12 (e.g. apt-get install python3.12 python3.12-venv) and retry" >&2
  exit 1
fi

echo "python3.12: $(command -v python3.12)"
python3.12 --version

if [ -d .venv ]; then
  rm -rf .venv
fi
python3.12 -m venv .venv
# shellcheck disable=SC1091
. .venv/bin/activate

echo "which python: $(command -v python)"
echo "which pip:    $(command -v pip || true)"
python -V
python -m pip --version

python - <<'PY'
import sys
assert sys.version_info[:2] == (3, 12), sys.version
print("interpreter ok:", sys.executable)
PY

python -m pip install --upgrade pip
python -m pip install -e python

python - <<'PY'
import sys
assert sys.version_info[:2] == (3, 12), sys.version
print("venv ready:", sys.executable)
PY
