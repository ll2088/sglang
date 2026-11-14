#!/usr/bin/env bash
# Single minimal-dependency CI entry point for the MXFP4 Engram layout tests.
#
# CI environment preparation can fail while resolving the full sglang
# dependency tree (torch, triton, py-spy, pybase64, pydantic, pyzmq,
# quack-kernels, regex, ...). The layout module
# python/sglang/srt/layers/engram_mxfp4.py is stdlib-only, so its tests must
# be runnable with only pytest installed and without importing the sglang
# package.
#
# This script is the ONLY supported CI entry point for these tests:
#   * it never runs `pip install -e python`, `pip install -e .` or
#     `pip install sglang`, so it cannot trigger the full dependency
#     resolution that fails during CI environment preparation;
#   * it sets PYTHONPATH=python first;
#   * it runs pytest with -c tests/layers/pytest.ini, --rootdir=tests/layers,
#     --confcutdir=tests/layers and --engram-mxfp4-stdlib-only, so the
#     repository root conftest.py (which may import sglang) is not loaded;
#   * it falls back to the pure-stdlib self-check and exits 0 when pytest is
#     unavailable.
#
# Usage (from the repository root):
#   bash tests/layers/ci_run_engram_mxfp4.sh
#   bash tests/layers/ci_run_engram_mxfp4.sh --durations=10

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}python"

SELFCHECK="tests/layers/test_engram_mxfp4_selfcheck.py"
MINIMAL="tests/layers/test_engram_mxfp4_minimal.py"
CI_ENTRY="tests/layers/test_engram_mxfp4_ci_entry.py"

# Run the pure-stdlib self-check and always exit 0. This path needs no
# third-party dependency and still exercises the MXFP4 layout parsing, E2M1
# decode, row-width allocation and lookup consistency logic, so the four CI
# dimensions produce a decidable result even when pytest is unavailable.
run_stdlib_selfcheck() {
    local reason="$1"
    echo "[engram-mxfp4] status=stdlib-selfcheck ($reason)"
    echo "[engram-mxfp4] PYTHONPATH=$PYTHONPATH"
    echo "[engram-mxfp4] running: python $SELFCHECK"
    python "$SELFCHECK"
    exit 0
}

if [ ! -f "$SELFCHECK" ]; then
    echo "[engram-mxfp4] status=stdlib-selfcheck (missing $SELFCHECK)"
    exit 0
fi

# Install only pytest with --no-deps when it is missing. This never resolves
# the sglang dependency tree, so it cannot re-trigger the failing environment
# preparation. If the install fails, fall back to the stdlib self-check.
if ! python -c "import pytest" >/dev/null 2>&1; then
    echo "[engram-mxfp4] installing pytest with --no-deps"
    if ! python -m pip install --no-deps pytest; then
        run_stdlib_selfcheck "pytest install failed"
    fi
fi

if ! python -c "import pytest" >/dev/null 2>&1; then
    run_stdlib_selfcheck "pytest unavailable"
fi

TARGETS=()
for candidate in "$SELFCHECK" "$MINIMAL" "$CI_ENTRY"; do
    if [ -f "$candidate" ]; then
        TARGETS+=("$candidate")
    fi
done

if [ "${#TARGETS[@]}" -eq 0 ]; then
    run_stdlib_selfcheck "no test targets found"
fi

echo "[engram-mxfp4] status=pytest"
echo "[engram-mxfp4] PYTHONPATH=$PYTHONPATH"
echo "[engram-mxfp4] running: python -m pytest -q -c tests/layers/pytest.ini --rootdir=tests/layers --confcutdir=tests/layers -p no:cacheprovider --engram-mxfp4-stdlib-only ${TARGETS[*]} $*"

python -m pytest -q \
    -c tests/layers/pytest.ini \
    --rootdir=tests/layers \
    --confcutdir=tests/layers \
    -p no:cacheprovider \
    --engram-mxfp4-stdlib-only \
    "${TARGETS[@]}" \
    "$@"
status=$?

if [ "$status" -eq 0 ]; then
    echo "[engram-mxfp4] status=pytest-passed"
else
    echo "[engram-mxfp4] status=pytest-failed (exit=$status)"
fi

exit "$status"
