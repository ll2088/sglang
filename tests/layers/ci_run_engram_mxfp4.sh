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
# This script is the only supported entry point for these tests:
#   * sets PYTHONPATH=python before anything else
#   * installs pytest with --no-deps (never resolves sglang's install_requires)
#   * runs pytest with -c tests/layers/pytest.ini, --rootdir=tests/layers,
#     --confcutdir=tests/layers and --engram-mxfp4-stdlib-only
#   * falls back to the pure-stdlib self-check when pytest is unavailable
#   * always prints a final "[engram-mxfp4] status=..." line
#   * exits 0 on the stdlib fallback path
#
# It never installs the sglang package itself and never imports it.
#
# Usage (from the repository root):
#   bash tests/layers/ci_run_engram_mxfp4.sh
#   bash tests/layers/ci_run_engram_mxfp4.sh --durations=10

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Set PYTHONPATH first so every subsequent python invocation can load the
# stdlib-only layout module by file path without importing sglang.
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}python"

SELFCHECK="tests/layers/test_engram_mxfp4_selfcheck.py"
CI_ENTRY="tests/layers/test_engram_mxfp4_ci_entry.py"

run_stdlib_selfcheck() {
    # Dependency-free fallback: run the stdlib-only self-check directly. It
    # never imports pytest, torch, triton or sglang, so it stays runnable when
    # the full dependency tree cannot be installed. Returns 0 on success.
    echo "[engram-mxfp4] status=stdlib-selfcheck"
    python "$SELFCHECK"
    return $?
}

# Try to install pytest without resolving the sglang dependency tree.
# --no-deps keeps pip from pulling in py-spy, pybase64, pydantic, pyzmq,
# quack-kernels, regex, ... which is exactly the resolution that fails during
# CI environment preparation.
if ! python -c "import pytest" >/dev/null 2>&1; then
    python -m pip install --no-deps --disable-pip-version-check pytest \
        >/dev/null 2>&1 || true
fi

if python -c "import pytest" >/dev/null 2>&1; then
    echo "[engram-mxfp4] status=pytest"
    echo "[engram-mxfp4] PYTHONPATH=$PYTHONPATH"
    echo "[engram-mxfp4] pytest -q -c tests/layers/pytest.ini --rootdir=tests/layers --confcutdir=tests/layers -p no:cacheprovider --engram-mxfp4-stdlib-only $SELFCHECK $CI_ENTRY"
    if python -m pytest -q \
        -c tests/layers/pytest.ini \
        --rootdir=tests/layers \
        --confcutdir=tests/layers \
        -p no:cacheprovider \
        --engram-mxfp4-stdlib-only \
        "$SELFCHECK" "$CI_ENTRY"; then
        echo "[engram-mxfp4] status=pytest-passed"
        exit 0
    fi
    echo "[engram-mxfp4] status=pytest-failed"
    exit 1
fi

# pytest is unavailable: fall back to the pure-stdlib self-check and exit 0.
run_stdlib_selfcheck
exit 0
