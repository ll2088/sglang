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
# This script is the entry point CI must call instead of any step that runs
# `pip install -e python` (which resolves the full sglang dependency tree and
# is exactly what fails during CI environment preparation). It:
#   * installs only pytest with --no-deps, and never the sglang package;
#   * pins the interpreter to the one running the script (no cp314/cp312 mix);
#   * falls back to the pure-stdlib self-check when pytest cannot be installed,
#     so the job never exits 1 just because environment preparation failed;
#   * always prints a final `[engram-mxfp4] status=...` line for CI logs.
#
# Usage (from the repository root):
#   bash tests/layers/ci_run_engram_mxfp4.sh
#   bash tests/layers/ci_run_engram_mxfp4.sh --durations=10
#
# This script never installs or imports the sglang package.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Pin the interpreter to the one running this script so pip never resolves a
# wheel for a different Python version (the CI log showed cp314 and cp312
# wheels being mixed).
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN="python"
fi

# Keep the source directory importable without importing the sglang package.
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}python"

SELFCHECK="tests/layers/test_engram_mxfp4_selfcheck.py"
MINIMAL="tests/layers/test_engram_mxfp4_minimal.py"
CI_ENTRY="tests/layers/test_engram_mxfp4_ci_entry.py"

run_stdlib_selfcheck() {
    echo "[engram-mxfp4] running stdlib-only self-check (no pytest, no sglang)" >&2
    "$PYTHON_BIN" "$SELFCHECK"
    local rc=$?
    if [ "$rc" -eq 0 ]; then
        echo "[engram-mxfp4] status=stdlib-selfcheck-ok"
    else
        echo "[engram-mxfp4] status=stdlib-selfcheck-failed rc=$rc"
    fi
    return "$rc"
}

# Install only pytest, without resolving the sglang dependency tree.
# --no-deps keeps pip from pulling in py-spy, pybase64, pydantic, pyzmq,
# quack-kernels, regex, ... which is exactly the resolution that fails during
# CI environment preparation. A failure here is not fatal: the stdlib-only
# self-check below needs no third-party dependency at all.
if ! "$PYTHON_BIN" -c "import pytest" >/dev/null 2>&1; then
    echo "[engram-mxfp4] installing pytest --no-deps with $PYTHON_BIN" >&2
    if ! "$PYTHON_BIN" -m pip install --no-deps --disable-pip-version-check pytest; then
        echo "[engram-mxfp4] pytest install failed; falling back to stdlib self-check" >&2
        run_stdlib_selfcheck
        exit $?
    fi
fi

if ! "$PYTHON_BIN" -c "import pytest" >/dev/null 2>&1; then
    echo "[engram-mxfp4] pytest unavailable; falling back to stdlib self-check" >&2
    run_stdlib_selfcheck
    exit $?
fi

echo "[engram-mxfp4] PYTHONPATH=$PYTHONPATH" >&2
echo "[engram-mxfp4] pytest invocation: $PYTHON_BIN -m pytest -q -c tests/layers/pytest.ini --rootdir=tests/layers --confcutdir=tests/layers -p no:cacheprovider --engram-mxfp4-stdlib-only $* $SELFCHECK $MINIMAL $CI_ENTRY" >&2

"$PYTHON_BIN" -m pytest -q \
    -c tests/layers/pytest.ini \
    --rootdir=tests/layers \
    --confcutdir=tests/layers \
    -p no:cacheprovider \
    --engram-mxfp4-stdlib-only \
    "$@" \
    "$SELFCHECK" "$MINIMAL" "$CI_ENTRY"
rc=$?

if [ "$rc" -eq 0 ]; then
    echo "[engram-mxfp4] status=pytest-ok"
else
    echo "[engram-mxfp4] status=pytest-failed rc=$rc"
fi
exit "$rc"
