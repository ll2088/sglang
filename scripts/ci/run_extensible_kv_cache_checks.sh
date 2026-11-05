#!/usr/bin/env bash
# Runs the four acceptance dimensions for the extensible KV cache port.
set -euxo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

VENV_PY="${REPO_ROOT}/.venv/bin/python"
if [ ! -x "${VENV_PY}" ]; then
  echo "ERROR: ${VENV_PY} missing; run scripts/ci/setup_extensible_kv_cache_env.sh first" >&2
  exit 1
fi

"${VENV_PY}" - <<'PY'
import sys
assert sys.version_info[:2] == (3, 12), sys.version
assert sys.prefix.endswith(".venv"), sys.prefix
print(sys.executable, sys.version, sys.prefix)
PY

TESTS=(
  tests/srt/mem_cache/test_vmm_allocator.py
  tests/srt/mem_cache/test_extensible_kv_cache.py
  tests/srt/arg_groups/test_arg_utils_extensible_kv_cache.py
)

echo "=== tests ==="
"${VENV_PY}" -m pytest -q "${TESTS[@]}"

echo "=== stability (3 runs, no random order) ==="
for i in 1 2 3; do
  "${VENV_PY}" -m pytest -q -p no:randomly "${TESTS[@]}"
done

echo "=== performance ==="
"${VENV_PY}" -m pytest -q benchmark --durations=10

echo "=== usability ==="
"${VENV_PY}" -m pytest -q test

echo "all dimensions passed"
