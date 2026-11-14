#!/usr/bin/env bash
# Single minimal-dependency CI entry point for the MXFP4 Engram layout tests.
#
# CI environment preparation can fail while resolving the full sglang
# dependency tree (torch, triton, py-spy, pybase64, pydantic, pyzmq,
# quack-kernels, regex, ...). The layout module
# python/sglang/srt/layers/engram_mxfp4.py is stdlib-only, so its tests only
# need pytest. This script installs pytest with --no-deps (never resolving the
# sglang package), runs pytest with the minimal stdlib-only config, and falls
# back to the pure-stdlib self-check when pytest cannot be installed.
#
# It never installs or imports the sglang package, so it cannot trigger the
# full dependency resolution that fails during CI environment preparation.
#
# Usage (from the repository root):
#   bash tests/layers/ci_run_engram_mxfp4.sh
#   bash tests/layers/ci_run_engram_mxfp4.sh --durations=10
#
# The script always prints a final `[engram-mxfp4] status=...` line so CI logs
# show whether the minimal pytest path or the stdlib fallback was used.

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

# Install only pytest, without resolving the sglang dependency tree.
# --no-deps keeps pip from pulling in py-spy, pybase64, pydantic, pyzmq,
# quack-kernels, regex, ... which is exactly the resolution that fails during
# CI environment preparation. Capture the exit code explicitly instead of
# relying on `set -e` (intentionally not enabled) so a failed download or
# resolve cannot terminate the script before the fallback runs.
python -m pip install --no-deps --disable-pip-version-check pytest
pip_status=$?
if [ "$pip_status" -ne 0 ]; then
    run_stdlib_selfcheck "pytest install failed (pip exit $pip_status)"
fi

# Sanity check: pytest must be importable. If not, fall back to the stdlib
# self-check instead of failing the job.
python -c "import pytest"
import_status=$?
if [ "$import_status" -ne 0 ]; then
    run_stdlib_selfcheck "pytest not importable (exit $import_status)"
fi

# Run pytest through the minimal, stdlib-only configuration. The
# --confcutdir/--rootdir flags keep the repository root conftest.py (which may
# import sglang) out of collection, and -p no:cacheprovider avoids cache
# writes in a read-only checkout.
echo "[engram-mxfp4] status=pytest"
echo "[engram-mxfp4] PYTHONPATH=$PYTHONPATH"
echo "[engram-mxfp4] running: python -m pytest -q -c tests/layers/pytest.ini --rootdir=tests/layers --confcutdir=tests/layers -p no:cacheprovider --engram-mxfp4-stdlib-only $SELFCHECK $MINIMAL $CI_ENTRY $*"
python -m pytest -q \
    -c tests/layers/pytest.ini \
    --rootdir=tests/layers \
    --confcutdir=tests/layers \
    -p no:cacheprovider \
    --engram-mxfp4-stdlib-only \
    "$SELFCHECK" "$MINIMAL" "$CI_ENTRY" "$@"
pytest_status=$?
if [ "$pytest_status" -ne 0 ]; then
    echo "[engram-mxfp4] status=pytest-failed (exit $pytest_status)"
    exit "$pytest_status"
fi

echo "[engram-mxfp4] status=pytest-passed"
exit 0
