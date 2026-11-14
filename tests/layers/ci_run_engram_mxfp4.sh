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
# This script is the entry point CI must call instead of running pytest from
# the repository root (which loads the root configuration and triggers the
# full dependency install). It:
#   * never runs ``pip install -e python`` and never resolves the sglang
#     package metadata;
#   * installs only pytest with ``--no-deps`` when pytest is missing;
#   * runs pytest with ``-c tests/layers/pytest.ini``,
#     ``--rootdir=tests/layers``, ``--confcutdir=tests/layers`` and
#     ``--engram-mxfp4-stdlib-only`` so the repository root conftest is not
#     loaded;
#   * falls back to the pure-stdlib self-check when pytest cannot be
#     installed, so the checks still execute and produce a result;
#   * always prints a final ``[engram-mxfp4] status=...`` line so CI logs show
#     which path was used.
#
# Usage (from the repository root):
#   bash tests/layers/ci_run_engram_mxfp4.sh
#   bash tests/layers/ci_run_engram_mxfp4.sh --durations=10
#
# This script never installs the sglang package itself and never imports it.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}python"

SELFCHECK="tests/layers/test_engram_mxfp4_selfcheck.py"
MINIMAL="tests/layers/test_engram_mxfp4_minimal.py"
CI_ENTRY="tests/layers/test_engram_mxfp4_ci_entry.py"

# Echo the exact commands so CI logs show the minimal-dependency path.
echo "[engram-mxfp4] repo_root=$REPO_ROOT"
echo "[engram-mxfp4] PYTHONPATH=$PYTHONPATH"

if ! python -c "import pytest" >/dev/null 2>&1; then
    echo "[engram-mxfp4] pytest missing; installing pytest --no-deps"
    # --no-deps keeps pip from resolving the sglang dependency tree
    # (py-spy, pybase64, pydantic, pyzmq, quack-kernels, regex, ...), which
    # is exactly the resolution that fails during CI environment preparation.
    python -m pip install --no-deps --disable-pip-version-check pytest \
        || echo "[engram-mxfp4] pytest install failed; will use stdlib fallback"
fi

if python -c "import pytest" >/dev/null 2>&1; then
    echo "[engram-mxfp4] running pytest minimal path"
    python -m pytest -q \
        -c tests/layers/pytest.ini \
        --rootdir=tests/layers \
        --confcutdir=tests/layers \
        -p no:cacheprovider \
        --engram-mxfp4-stdlib-only \
        "$@" \
        "$SELFCHECK" "$MINIMAL" "$CI_ENTRY"
    status=$?
    echo "[engram-mxfp4] status=pytest exit=$status"
    exit $status
fi

# pytest is not installed: run the stdlib-only self-check directly. This path
# never imports sglang/torch/triton, so it stays runnable when the full sglang
# dependency tree cannot be installed during CI preparation.
echo "[engram-mxfp4] pytest unavailable; running stdlib-only self-check"
python "$SELFCHECK"
status=$?
echo "[engram-mxfp4] status=stdlib exit=$status"
exit $status
