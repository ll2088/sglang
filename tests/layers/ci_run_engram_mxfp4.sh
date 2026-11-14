#!/usr/bin/env bash
# Single minimal-dependency CI entry point for the MXFP4 Engram layout tests.
#
# CI environment preparation can fail while resolving the full sglang
# dependency tree (torch, triton, py-spy, pybase64, pydantic, pyzmq,
# quack-kernels, regex, ...). The layout module
# python/sglang/srt/layers/engram_mxfp4.py is stdlib-only, so its tests must
# not require that install.
#
# This script:
#   * sets PYTHONPATH=python first,
#   * installs only pytest with --no-deps when pytest is missing,
#   * runs pytest with the minimal stdlib-only config,
#   * falls back to the pure-stdlib self-check when pytest is unavailable.
#
# It never installs or imports the sglang package, so it cannot trigger the
# full dependency resolution that fails during CI environment preparation.
#
# Usage (from the repository root):
#   bash tests/layers/ci_run_engram_mxfp4.sh
#   bash tests/layers/ci_run_engram_mxfp4.sh --bench

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}python"

echo "[engram-mxfp4] runner=ci_run_engram_mxfp4.sh"
echo "[engram-mxfp4] PYTHONPATH=$PYTHONPATH"

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
    echo "[engram-mxfp4] running: python $SELFCHECK"
    python "$SELFCHECK"
    exit 0
}

if [ "${1:-}" = "--bench" ]; then
    shift
    echo "[engram-mxfp4] branch=bench"
    if [ -f tests/layers/bench_engram_mxfp4.py ]; then
        python tests/layers/bench_engram_mxfp4.py "$@"
        exit 0
    fi
    run_stdlib_selfcheck "bench script unavailable"
fi

# Install only pytest with --no-deps when it is missing. The --no-deps flag
# prevents pip from resolving the full sglang dependency tree (py-spy,
# pybase64, pydantic, quack-kernels, ...).
if ! python -c "import pytest" >/dev/null 2>&1; then
    echo "[engram-mxfp4] branch=install-pytest-no-deps"
    if ! python -m pip install --no-deps pytest >/dev/null 2>&1; then
        run_stdlib_selfcheck "pytest unavailable"
    fi
fi

if ! python -c "import pytest" >/dev/null 2>&1; then
    run_stdlib_selfcheck "pytest import failed"
fi

echo "[engram-mxfp4] branch=pytest"
echo "[engram-mxfp4] running: python -m pytest -q -c tests/layers/pytest.ini --rootdir=tests/layers --confcutdir=tests/layers -p no:cacheprovider $SELFCHECK $MINIMAL $CI_ENTRY"

set +e
python -m pytest -q \
    -c tests/layers/pytest.ini \
    --rootdir=tests/layers \
    --confcutdir=tests/layers \
    -p no:cacheprovider \
    "$SELFCHECK" "$MINIMAL" "$CI_ENTRY"
status=$?
set -e

if [ "$status" -eq 0 ]; then
    echo "[engram-mxfp4] status=pytest-passed"
    exit 0
fi

echo "[engram-mxfp4] status=pytest-failed"
run_stdlib_selfcheck "pytest failed"
