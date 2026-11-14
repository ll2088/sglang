#!/usr/bin/env bash
# Single minimal-dependency CI entry point for the MXFP4 Engram tests.
#
# CI environment preparation can fail while resolving the full sglang
# dependency tree (torch, triton, py-spy, pybase64, pydantic, quack-kernels,
# regex, ...). The MXFP4 layout module
# python/sglang/srt/layers/engram_mxfp4.py is stdlib-only, so its tests only
# need pytest. This script:
#   * sets PYTHONPATH=python first,
#   * installs only pytest with --no-deps (never resolving sglang deps),
#   * runs pytest with the minimal stdlib-only config,
#   * falls back to the pure-stdlib self-check when pytest is unavailable,
#   * always prints a final `[engram-mxfp4] status=...` line and exits 0 on
#     the fallback path so a pip failure cannot fail the CI job.
#
# It never installs or imports the sglang package itself.
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
    local selfcheck_status=$?
    if [ "$selfcheck_status" -ne 0 ]; then
        echo "[engram-mxfp4] status=stdlib-selfcheck-failed (exit $selfcheck_status)"
        exit "$selfcheck_status"
    fi
    echo "[engram-mxfp4] status=stdlib-selfcheck-passed"
    exit 0
}

# If pytest is already importable, run the minimal pytest path directly.
# Otherwise try to install only pytest with --no-deps; a failure here must
# not fail the job, so the exit code is captured explicitly (set -e is
# intentionally not enabled) and the stdlib self-check runs instead.
if ! python -c "import pytest" >/dev/null 2>&1; then
    echo "[engram-mxfp4] pytest not importable; installing pytest with --no-deps"
    python -m pip install --no-deps --disable-pip-version-check pytest
    pip_status=$?
    if [ "$pip_status" -ne 0 ]; then
        echo "[engram-mxfp4] pytest install failed (pip exit $pip_status)"
        run_stdlib_selfcheck "pytest install failed (pip exit $pip_status)"
    fi
    if ! python -c "import pytest" >/dev/null 2>&1; then
        run_stdlib_selfcheck "pytest not importable after install"
    fi
fi

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
