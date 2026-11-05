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
# Single entry point for the extensible KV cache checks. It prepares the
# environment through setup_extensible_kv_cache_env.sh and then runs the
# tests, stability, performance and usability dimensions.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
PYTHON_BIN="${VENV_DIR}/bin/python"

log() {
    printf '[run_extensible_kv_cache_checks] %s\n' "$*" >&2
}

bash "${REPO_ROOT}/scripts/ci/setup_extensible_kv_cache_env.sh"

# Re-assert the interpreter after setup so a stale shell cannot leak in.
"${PYTHON_BIN}" - <<'PY' || exit 1
import sys

if sys.version_info[:2] != (3, 12):
    raise SystemExit(
        f"expected Python 3.12, got {sys.version.split()[0]} at {sys.executable}"
    )
PY

TEST_TARGETS=(
    "tests/srt/mem_cache/test_vmm_allocator.py"
    "tests/srt/mem_cache/test_extensible_kv_cache.py"
    "tests/srt/arg_groups/test_arg_utils_extensible_kv_cache.py"
)

log "tests dimension"
"${PYTHON_BIN}" -m pytest -q "${TEST_TARGETS[@]}"

log "stability dimension"
for i in 1 2 3; do
    "${PYTHON_BIN}" -m pytest -q -p no:randomly "${TEST_TARGETS[@]}" || exit 1
done

log "performance dimension"
"${PYTHON_BIN}" -m pytest -q benchmark --durations=10

log "usability dimension"
"${PYTHON_BIN}" -m pytest -q test

log "all dimensions passed"
