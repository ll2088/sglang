#!/usr/bin/env bash
# Run the extensible KV cache checks in the isolated venv. Any failure in the
# setup or in a dimension aborts the whole run.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="${REPO_ROOT}/.venv/bin/python"

bash "${REPO_ROOT}/scripts/setup_extensible_kv_cache_env.sh"

if [[ ! -x "${VENV_PYTHON}" ]]; then
    echo "error: venv python missing at ${VENV_PYTHON}" >&2
    exit 1
fi

cd "${REPO_ROOT}"

# tests
"${VENV_PYTHON}" -m pytest -q \
    tests/srt/mem_cache/test_vmm_allocator.py \
    tests/srt/mem_cache/test_extensible_kv_cache.py \
    tests/srt/arg_groups/test_arg_utils_extensible_kv_cache.py

# stability: repeat without random ordering to catch order dependence
for _ in 1 2 3; do
    "${VENV_PYTHON}" -m pytest -q -p no:randomly \
        tests/srt/mem_cache/test_vmm_allocator.py \
        tests/srt/mem_cache/test_extensible_kv_cache.py \
        tests/srt/arg_groups/test_arg_utils_extensible_kv_cache.py
done

# performance
"${VENV_PYTHON}" -m pytest -q benchmark --durations=10

# usability
"${VENV_PYTHON}" -m pytest -q test
