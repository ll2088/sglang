#!/usr/bin/env bash
# Single minimal-dependency CI entry point for the MXFP4 Engram layout tests.
#
# CI environment preparation can fail while resolving the full sglang
# dependency tree (torch, triton, py-spy, pybase64, pydantic, pyzmq,
# quack-kernels, regex, ...). The layout module
# python/sglang/srt/layers/engram_mxfp4.py is stdlib-only, so its tests must be
# runnable with only pytest installed and without importing the sglang package.
#
# This script is the one command CI should call instead of any step that runs
# `pip install -e python` (or otherwise resolves the sglang package). It:
#   1. sets PYTHONPATH to the repository python/ directory first, so nothing
#      later needs to import the sglang package to find the layout module;
#   2. installs only pytest with --no-deps, so pip never resolves sglang's
#      install_requires (py-spy, pybase64, pydantic, quack-kernels, ...);
#   3. runs pytest with -c tests/layers/pytest.ini, --rootdir=tests/layers,
#      --confcutdir=tests/layers and --engram-mxfp4-stdlib-only, so the
#      repository root conftest.py (which may import sglang) is not loaded;
#   4. falls back to the pure-stdlib self-check when pytest cannot be
#      installed, so the layout / E2M1 decode / row-width allocation checks
#      still run in a minimal environment.
#
# It always prints a final `[engram-mxfp4] status=...` line so CI logs show
# which path was taken, and echoes the resolved PYTHONPATH and the exact
# pytest invocation so the minimal-dependency path is verifiable.
#
# Usage (from the repository root):
#   bash tests/layers/ci_run_engram_mxfp4.sh
#   bash tests/layers/ci_run_engram_mxfp4.sh --durations=10
#
# This script never installs the sglang package and never imports it.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Step 1: make the layout module importable by path before anything else, so
# no later step needs to resolve the sglang package to find it.
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}python"
echo "[engram-mxfp4] PYTHONPATH=$PYTHONPATH"

SELFCHECK="tests/layers/test_engram_mxfp4_selfcheck.py"
MINIMAL="tests/layers/test_engram_mxfp4_minimal.py"
CI_ENTRY="tests/layers/test_engram_mxfp4_ci_entry.py"

run_stdlib_selfcheck() {
    echo "[engram-mxfp4] status=stdlib-selfcheck"
    exec python "$SELFCHECK" "$@"
}

# Step 2: install only pytest, without resolving the sglang dependency tree.
# --no-deps keeps pip from pulling in py-spy, pybase64, pydantic,
# quack-kernels, regex, ... which is exactly the resolution that fails during
# CI environment preparation. If pytest is already importable, skip the
# install entirely so no network access is required.
if ! python -c "import pytest" >/dev/null 2>&1; then
    echo "[engram-mxfp4] installing pytest with --no-deps"
    if ! python -m pip install --no-deps --disable-pip-version-check pytest; then
        echo "[engram-mxfp4] pytest install failed; falling back to stdlib self-check"
        run_stdlib_selfcheck "$@"
    fi
fi

if ! python -c "import pytest" >/dev/null 2>&1; then
    echo "[engram-mxfp4] pytest unavailable; falling back to stdlib self-check"
    run_stdlib_selfcheck "$@"
fi

# Step 3: run pytest with the minimal, stdlib-only configuration. The
# --confcutdir / --rootdir / -c flags keep the repository root conftest.py
# (which may import sglang) out of collection, so no sglang/torch/triton
# import is triggered.
echo "[engram-mxfp4] status=pytest"
echo "[engram-mxfp4] pytest invocation: python -m pytest -q -c tests/layers/pytest.ini --rootdir=tests/layers --confcutdir=tests/layers -p no:cacheprovider --engram-mxfp4-stdlib-only $* $SELFCHECK $MINIMAL $CI_ENTRY"

if python -m pytest -q \
    -c tests/layers/pytest.ini \
    --rootdir=tests/layers \
    --confcutdir=tests/layers \
    -p no:cacheprovider \
    --engram-mxfp4-stdlib-only \
    "$@" \
    "$SELFCHECK" "$MINIMAL" "$CI_ENTRY"; then
    echo "[engram-mxfp4] status=pytest-passed"
    exit 0
fi

# Step 4: pytest ran but failed. Do not silently fall back to the stdlib
# self-check here: a real test failure must surface as a failure. The stdlib
# fallback is only for the case where pytest cannot be installed at all.
echo "[engram-mxfp4] status=pytest-failed"
exit 1
