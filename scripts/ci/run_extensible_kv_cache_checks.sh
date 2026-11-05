#!/usr/bin/env bash
# Thin wrapper that prepares the isolated environment and then runs the
# extensible KV cache checks. The exit code of the inner command is propagated
# so CI reports the real failure.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

bash "${REPO_ROOT}/scripts/ci/setup_extensible_kv_cache_env.sh"

PY="${REPO_ROOT}/.venv/bin/python"

if [ "$#" -eq 0 ]; then
    set -- -m pytest -q \
        tests/srt/mem_cache/test_vmm_allocator.py \
        tests/srt/mem_cache/test_extensible_kv_cache.py \
        tests/srt/arg_groups/test_arg_utils_extensible_kv_cache.py
fi

exec "${PY}" "$@"
