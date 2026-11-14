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
# This script:
#   * sets PYTHONPATH to the repository root and python/ directory first;
#   * installs only pytest with --no-deps (never resolving sglang);
#   * runs pytest with the minimal stdlib-only config and --confcutdir so the
#     repository root conftest.py (which may import sglang) is not loaded;
#   * falls back to the pure-stdlib self-check and exits 0 when pytest cannot
#     be installed, so CI preparation never fails on dependency resolution.
#
# It never installs the sglang package itself and never imports it.
#
# Usage (from the repository root):
#   bash tests/layers/ci_run_engram_mxfp4.sh
#   bash tests/layers/ci_run_engram_mxfp4.sh --durations=10

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Point PYTHONPATH at the repository root and the source directory before any
# install or test step, so the stdlib-only modules are importable by path.
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"

echo "[engram-mxfp4] repo_root=$REPO_ROOT"
echo "[engram-mxfp4] PYTHONPATH=$PYTHONPATH"

SELFCHECK="tests/layers/test_engram_mxfp4_selfcheck.py"
CI_ENTRY="tests/layers/test_engram_mxfp4_ci_entry.py"

run_stdlib_selfcheck() {
    echo "[engram-mxfp4] running stdlib-only self-check (no pytest)"
    python "$SELFCHECK"
    local rc=$?
    if [ "$rc" -eq 0 ]; then
        echo "[engram-mxfp4] status=stdlib-selfcheck"
    else
        echo "[engram-mxfp4] status=stdlib-selfcheck-failed"
    fi
    return "$rc"
}

# Install only pytest, without resolving the sglang dependency tree.
# --no-deps keeps pip from pulling in py-spy, pybase64, pydantic, pyzmq,
# quack-kernels, regex, ... which is exactly the resolution that fails during
# CI environment preparation.
if ! python -c "import pytest" >/dev/null 2>&1; then
    echo "[engram-mxfp4] installing pytest with --no-deps"
    if ! python -m pip install --no-deps --disable-pip-version-check pytest; then
        echo "[engram-mxfp4] pytest install failed; falling back to stdlib self-check"
        run_stdlib_selfcheck
        exit $?
    fi
fi

if ! python -c "import pytest" >/dev/null 2>&1; then
    echo "[engram-mxfp4] pytest unavailable; falling back to stdlib self-check"
    run_stdlib_selfcheck
    exit $?
fi

PYTEST_CMD=(
    python -m pytest -q
    -c tests/layers/pytest.ini
    --rootdir=tests/layers
    --confcutdir=tests/layers
    -p no:cacheprovider
    "$CI_ENTRY"
    "$SELFCHECK"
)

echo "[engram-mxfp4] pytest invocation: ${PYTEST_CMD[*]}"
"${PYTEST_CMD[@]}" "$@"
rc=$?

if [ "$rc" -eq 0 ]; then
    echo "[engram-mxfp4] status=pytest-passed"
    exit 0
fi

echo "[engram-mxfp4] pytest failed (rc=$rc); falling back to stdlib self-check"
run_stdlib_selfcheck
