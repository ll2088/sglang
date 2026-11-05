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
# Single entry point for the extensible KV cache checks. Runs the four
# dimensions (tests, stability, performance, usability) and exits non-zero on
# the first failure.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"

assert_venv_python() {
  "${PYTHON_BIN}" - <<'PY'
import sys
assert sys.version_info[:2] == (3, 12), sys.version
assert sys.prefix.endswith(".venv"), sys.prefix
PY
}

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "missing ${PYTHON_BIN}; run scripts/ci/setup_extensible_kv_cache_env.sh first" >&2
  exit 1
fi

TEST_FILES=(
  "tests/srt/mem_cache/test_vmm_allocator.py"
  "tests/srt/mem_cache/test_extensible_kv_cache.py"
  "tests/srt/arg_groups/test_arg_utils_extensible_kv_cache.py"
)

cd "${REPO_ROOT}"

assert_venv_python
echo "== tests =="
"${PYTHON_BIN}" -m pytest -q "${TEST_FILES[@]}"

assert_venv_python
echo "== stability =="
for i in 1 2 3; do
  "${PYTHON_BIN}" -m pytest -q -p no:randomly "${TEST_FILES[@]}"
done

assert_venv_python
echo "== performance =="
"${PYTHON_BIN}" -m pytest -q benchmark --durations=10

assert_venv_python
echo "== usability =="
"${PYTHON_BIN}" -m pytest -q test

echo "all extensible KV cache checks passed"
