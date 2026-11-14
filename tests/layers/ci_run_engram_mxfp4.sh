#!/usr/bin/env bash
# Single minimal-dependency CI entry point for the MXFP4 Engram layout tests.
#
# CI environment preparation can fail while resolving the full sglang
# dependency tree (torch, triton, py-spy, pybase64, pydantic, pyzmq,
# quack-kernels, regex, ...). The layout module
# python/sglang/srt/layers/engram_mxfp4.py is stdlib-only, so its tests must be
# runnable without importing the sglang package and without resolving that
# dependency tree.
#
# This script NEVER installs anything. Installing pytest would still risk
# resolving the full sglang dependency tree during CI environment preparation,
# which is exactly the failure this entry point exists to avoid. When pytest is
# unavailable it falls back to the pure-stdlib self-check, which needs no
# third-party dependency at all, and exits 0.
#
# It sets PYTHONPATH=python first, runs pytest with the minimal stdlib-only
# config (-c tests/layers/pytest.ini, --rootdir=tests/layers,
# --confcutdir=tests/layers, --engram-mxfp4-stdlib-only) when pytest is
# available, and otherwise runs the stdlib self-check directly.
#
# Usage (from the repository root):
#   bash tests/layers/ci_run_engram_mxfp4.sh
#   bash tests/layers/ci_run_engram_mxfp4.sh --bench
#
# It always prints a final `[engram-mxfp4] status=...` line so CI logs show
# which path was taken.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}python"

SELFCHECK="tests/layers/test_engram_mxfp4_selfcheck.py"
BENCH="tests/layers/bench_engram_mxfp4.py"

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

# Benchmark mode: run the dependency-free benchmark when present, otherwise
# fall back to the stdlib self-check. Never installs anything.
if [ "${1:-}" = "--bench" ]; then
    shift
    if [ -f "$BENCH" ]; then
        echo "[engram-mxfp4] status=bench"
        echo "[engram-mxfp4] PYTHONPATH=$PYTHONPATH"
        echo "[engram-mxfp4] running: python $BENCH"
        python "$BENCH"
        exit $?
    fi
    run_stdlib_selfcheck "missing $BENCH"
fi

# pytest is optional. When it is unavailable, fall back to the stdlib
# self-check instead of installing anything.
if ! python -c "import pytest" >/dev/null 2>&1; then
    run_stdlib_selfcheck "pytest unavailable"
fi

echo "[engram-mxfp4] status=pytest"
echo "[engram-mxfp4] PYTHONPATH=$PYTHONPATH"
echo "[engram-mxfp4] running: python -m pytest -q -c tests/layers/pytest.ini --rootdir=tests/layers --confcutdir=tests/layers --engram-mxfp4-stdlib-only -p no:cacheprovider tests/layers/test_engram_mxfp4_selfcheck.py tests/layers/test_engram_mxfp4.py"

python -m pytest -q \
    -c tests/layers/pytest.ini \
    --rootdir=tests/layers \
    --confcutdir=tests/layers \
    --engram-mxfp4-stdlib-only \
    -p no:cacheprovider \
    tests/layers/test_engram_mxfp4_selfcheck.py \
    tests/layers/test_engram_mxfp4.py
status=$?

if [ "$status" -eq 0 ]; then
    echo "[engram-mxfp4] status=pytest-passed"
    exit 0
fi

echo "[engram-mxfp4] status=pytest-failed"
exit "$status"
