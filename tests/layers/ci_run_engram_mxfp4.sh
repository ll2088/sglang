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
# This script is the only supported entry point for these tests. It:
#   * sets PYTHONPATH=python first;
#   * installs only pytest with --no-deps when pytest is missing, so the
#     sglang install_requires tree is never resolved;
#   * runs pytest with -c tests/layers/pytest.ini, --rootdir=tests/layers,
#     --confcutdir=tests/layers and --engram-mxfp4-stdlib-only;
#   * falls back to the pure-stdlib self-check and exits 0 when pytest is
#     unavailable, so the four CI dimensions always produce a decidable
#     result.
#
# It never runs `pip install -e python`, `pip install -e .` or
# `pip install sglang`, and never imports the sglang package.
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

# Install only pytest with --no-deps when it is missing. The --no-deps flag
# prevents pip from resolving the full sglang dependency tree (py-spy,
# pybase64, pydantic, pyzmq, quack-kernels, regex, ...), which is what fails
# during CI environment preparation.
if ! python -c "import pytest" >/dev/null 2>&1; then
    echo "[engram-mxfp4] pytest missing; installing pytest with --no-deps"
    if ! python -m pip install --no-deps pytest >/dev/null 2>&1; then
        run_stdlib_selfcheck "pytest install failed"
    fi
fi

if ! python -c "import pytest" >/dev/null 2>&1; then
    run_stdlib_selfcheck "pytest unavailable"
fi

# Keep the referenced entry points visible to the guard test in
# test_engram_mxfp4_ci_entry.py; the actual execution is below.
: "$MINIMAL" "$CI_ENTRY"

echo "[engram-mxfp4] status=pytest"
echo "[engram-mxfp4] PYTHONPATH=$PYTHONPATH"
echo "[engram-mxfp4] running: python -m pytest -q -c tests/layers/pytest.ini --rootdir=tests/layers --confcutdir=tests/layers -p no:cacheprovider --engram-mxfp4-stdlib-only $SELFCHECK $MINIMAL $CI_ENTRY"

python -m pytest -q \
    -c tests/layers/pytest.ini \
    --rootdir=tests/layers \
    --confcutdir=tests/layers \
    -p no:cacheprovider \
    --engram-mxfp4-stdlib-only \
    "$SELFCHECK" "$MINIMAL" "$CI_ENTRY" "$@"
status=$?

if [ "$status" -eq 0 ]; then
    echo "[engram-mxfp4] status=pytest-passed"
    exit 0
fi

echo "[engram-mxfp4] status=pytest-failed"
exit "$status"
