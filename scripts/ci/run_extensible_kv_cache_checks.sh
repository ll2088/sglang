#!/usr/bin/env bash
# Unique check entry for the extensible KV cache port.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

bash "${REPO_ROOT}/scripts/ci/setup_extensible_kv_cache_env.sh"

VENV_PY="${REPO_ROOT}/.venv/bin/python"

"${VENV_PY}" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'

TEST_FILES=(
  tests/srt/mem_cache/test_vmm_allocator.py
  tests/srt/mem_cache/test_extensible_kv_cache.py
  tests/srt/arg_groups/test_arg_utils_extensible_kv_cache.py
)

"${VENV_PY}" -m pytest -q "${TEST_FILES[@]}"

for _ in 1 2 3; do
  "${VENV_PY}" -m pytest -q -p no:randomly "${TEST_FILES[@]}"
done
