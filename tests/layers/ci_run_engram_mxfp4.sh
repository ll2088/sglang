#!/usr/bin/env bash
# Single minimal-dependency CI entry point for the MXFP4 Engram layout tests.
#
# CI environment preparation can fail while resolving the full sglang
# dependency tree (torch, triton, py-spy, pybase64, pydantic, quack-kernels,
# regex, ...). The layout module python/sglang/srt/layers/engram_mxfp4.py is
# stdlib-only, so its tests only need pytest. This script:
#   * sets PYTHONPATH=python first,
#   * installs only pytest with --no-deps (never resolving sglang),
#   * runs pytest with the minimal stdlib-only config,
#   * falls back to the pure-stdlib self-check when pytest is unavailable,
#   * always exits 0 on the stdlib fallback path,
#   * prints a final [engram-mxfp4] status=... line for CI logs.
#
# It never installs or imports the sglang package, so it cannot trigger the
# full dependency resolution that fails during CI environment preparation.
#
# Usage (from the repository root):
#   bash tests/layers/ci_run_engram_mxfp4.sh
#   bash tests/layers/ci_run_engram_mxfp4.sh --durations=10

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Set PYTHONPATH before any install or test invocation so the stdlib-only
# layout module is importable by file path without the sglang package.
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}python"

echo "[engram-mxfp4] PYTHONPATH=$PYTHONPATH"

SELFCHECK="tests/layers/test_engram_mxfp4_selfcheck.py"
CI_ENTRY="tests/layers/test_engram_mxfp4_ci_entry.py"

run_stdlib_selfcheck() {
    echo "[engram-mxfp4] running stdlib self-check (no pytest)"
    if python "$SELFCHECK"; then
        echo "[engram-mxfp4] status=stdlib-selfcheck"
        exit 0
    fi
    echo "[engram-mxfp4] status=stdlib-selfcheck-failed"
    exit 1
}

# Install only pytest, without resolving the sglang dependency tree.
# --no-deps keeps pip from pulling in py-spy, pybase64, pydantic,
# quack-kernels, regex, ... which is exactly the resolution that fails during
# CI environment preparation. If pytest is already importable, skip install.
if ! python -c "import pytest" >/dev/null 2>&1; then
    echo "[engram-mxfp4] installing pytest with --no-deps"
    if ! python -m pip install --no-deps --disable-pip-version-check pytest; then
        echo "[engram-mxfp4] pytest install failed; falling back to stdlib self-check"
        run_stdlib_selfcheck
    fi
fi

if ! python -c "import pytest" >/dev/null 2>&1; then
    echo "[engram-mxfp4] pytest unavailable; falling back to stdlib self-check"
    run_stdlib_selfcheck
fi

PYTEST_CMD=(python -m pytest -q \
    -c tests/layers/pytest.ini \
    --rootdir=tests/layers \
    --confcutdir=tests/layers \
    -p no:cacheprovider \
    --engram-mxfp4-stdlib-only \
    "$SELFCHECK" "$CI_ENTRY")

echo "[engram-mxfp4] pytest invocation: ${PYTEST_CMD[*]} $*"

if "${PYTEST_CMD[@]}" "$@"; then
    echo "[engram-mxfp4] status=pytest-passed"
    exit 0
fi

echo "[engram-mxfp4] status=pytest-failed"
exit 1
