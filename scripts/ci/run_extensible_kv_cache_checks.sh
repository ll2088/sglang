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
# End-to-end entry point for the extensible KV cache checks.
#
# Sources the single setup implementation, then runs the four dimensions
# (tests, stability, performance, usability) with the venv interpreter.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck source=scripts/ci/setup_extensible_kv_cache_env.sh
source "${SCRIPT_DIR}/setup_extensible_kv_cache_env.sh"

setup_extensible_kv_cache_env

PY="${VENV_DIR}/bin/python"

cd "${REPO_ROOT}"

run_tests() {
    "${PY}" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
    "${PY}" -m pytest -q \
        tests/srt/mem_cache/test_vmm_allocator.py \
        tests/srt/mem_cache/test_extensible_kv_cache.py \
        tests/srt/arg_groups/test_arg_utils_extensible_kv_cache.py
}

run_stability() {
    "${PY}" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
    local i
    for i in 1 2 3; do
        "${PY}" -m pytest -q -p no:randomly \
            tests/srt/mem_cache/test_vmm_allocator.py \
            tests/srt/mem_cache/test_extensible_kv_cache.py \
            tests/srt/arg_groups/test_arg_utils_extensible_kv_cache.py \
            || exit 1
    done
}

run_performance() {
    "${PY}" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
    "${PY}" -m pytest -q benchmark --durations=10
}

run_usability() {
    "${PY}" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
    "${PY}" -m pytest -q test
}

case "${1:-all}" in
    tests) run_tests ;;
    stability) run_stability ;;
    performance) run_performance ;;
    usability) run_usability ;;
    all)
        run_tests
        run_stability
        run_performance
        run_usability
        ;;
    *)
        echo "unknown dimension: $1" >&2
        exit 2
        ;;
esac
