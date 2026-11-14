#!/usr/bin/env bash
# Dependency-free CI entry point for the MXFP4 Engram layout tests.
#
# CI environment preparation can fail while resolving the full sglang
# dependency tree (torch, triton, py-spy, pybase64, pydantic, quack-kernels,
# regex, ...). The layout module python/sglang/srt/layers/engram_mxfp4.py is
# stdlib-only, so its tests must be runnable with only pytest installed and
# without importing the sglang package.
#
# This script is the single command CI should call for the MXFP4 layout
# tests. It never installs the sglang package and never imports it:
#   * it installs only pytest (with --no-deps) when pytest is missing;
#   * it runs pytest with -c tests/layers/pytest.ini, --rootdir=tests/layers
#     and --confcutdir=tests/layers so the repository root conftest.py (which
#     may import sglang) is not loaded;
#   * it passes --engram-mxfp4-stdlib-only so the tests/layers conftest keeps
#     the sglang package out of sys.modules;
#   * if pytest cannot be installed, it falls back to the pure-stdlib
#     self-check, which needs no third-party dependency at all.
#
# Usage (from the repository root):
#   bash tests/layers/ci_run_engram_mxfp4.sh
#   bash tests/layers/ci_run_engram_mxfp4.sh --durations=10
#
# The script always prints a final, machine-checkable status line so CI logs
# show whether the minimal path or the stdlib fallback was used.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}python"

SELFCHECK="tests/layers/test_engram_mxfp4_selfcheck.py"
MINIMAL="tests/layers/test_engram_mxfp4_minimal.py"
CI_ENTRY="tests/layers/test_engram_mxfp4_ci_entry.py"

# Install only pytest, without resolving the sglang dependency tree. This is
# the step that replaces ``pip install -e python`` for these tests; it must
# never pull in py-spy, pyzmq, quack-kernels, regex, ... which is exactly the
# resolution that fails during CI environment preparation.
if ! python -c "import pytest" >/dev/null 2>&1; then
    echo "[engram-mxfp4] pytest missing; installing pytest only (--no-deps)" >&2
    if ! python -m pip install --no-deps --disable-pip-version-check pytest; then
        echo "[engram-mxfp4] pytest install failed; falling back to stdlib self-check" >&2
        python "$SELFCHECK"
        echo "[engram-mxfp4] status=stdlib-fallback"
        exit 0
    fi
fi

if python -c "import pytest" >/dev/null 2>&1; then
    echo "[engram-mxfp4] running pytest with minimal config" >&2
    python -m pytest -q \
        -c tests/layers/pytest.ini \
        --rootdir=tests/layers \
        --confcutdir=tests/layers \
        -p no:cacheprovider \
        --engram-mxfp4-stdlib-only \
        "$@" \
        "$SELFCHECK" "$MINIMAL" "$CI_ENTRY"
    echo "[engram-mxfp4] status=pytest-minimal"
    exit 0
fi

# pytest is not installed and could not be installed: run the stdlib-only
# self-check directly. This path never imports sglang/torch/triton, so it
# stays runnable when the full sglang dependency tree cannot be installed.
echo "[engram-mxfp4] pytest unavailable; running stdlib-only self-check" >&2
python "$SELFCHECK"
echo "[engram-mxfp4] status=stdlib-fallback"
