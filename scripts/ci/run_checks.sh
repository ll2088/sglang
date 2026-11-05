#!/usr/bin/env bash
# Run the four verification dimensions against the environment built by
# scripts/ci/setup_env.sh. All invocations go through .venv/bin/python -m ...
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

unset PYTHONPATH || true
unset PYTHONHOME || true
export PYTHONNOUSERSITE=1

VENV_PY=".venv/bin/python"
if [ ! -x "$VENV_PY" ]; then
  echo "error: $VENV_PY missing; run scripts/ci/setup_env.sh first" >&2
  exit 1
fi

"$VENV_PY" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'

TARGETS=(
  tests/srt/mem_cache/test_vmm_allocator.py
  tests/srt/mem_cache/test_extensible_kv_cache.py
  tests/srt/arg_groups/test_arg_utils_extensible_kv_cache.py
)

run_tests() {
  "$VENV_PY" -m pytest -q "$@"
}

echo "== tests =="
run_tests "${TARGETS[@]}"

echo "== stability =="
for i in 1 2 3; do
  echo "-- run $i --"
  run_tests -p no:randomly "${TARGETS[@]}"
done

echo "== performance =="
run_tests benchmark --durations=10

echo "== usability =="
run_tests test

echo "run_checks.sh: all dimensions passed"
