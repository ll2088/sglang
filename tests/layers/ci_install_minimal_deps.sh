#!/usr/bin/env bash
# Minimal-dependency preparation for the MXFP4 Engram layout tests.
#
# CI environment preparation can fail while resolving the full sglang
# dependency tree (torch, triton, py-spy, pybase64, pydantic, pyzmq,
# quack-kernels, regex, ...). The layout module
# python/sglang/srt/layers/engram_mxfp4.py is stdlib-only, so its tests must
# not require that install.
#
# This script never installs the sglang package and never runs
# `pip install -e python` / `pip install -e .` / `pip install sglang`; those
# commands resolve the full dependency tree and re-trigger the failing CI
# environment preparation. It delegates to the single minimal-dependency CI
# entry point, and when that entry point is missing (for example because CI
# environment preparation failed before the checkout was complete) it
# delegates to the pure-stdlib entry point, which installs nothing and always
# exits 0.
#
# Usage (from the repository root):
#   bash tests/layers/ci_install_minimal_deps.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}python"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN="python"
fi

# Print the branch actually taken so CI logs make it verifiable that the
# minimal-dependency path (and not `pip install -e python`) was used.
echo "[engram-mxfp4] runner=ci_install_minimal_deps.sh"
echo "[engram-mxfp4] python=$PYTHON_BIN ($(command -v "$PYTHON_BIN" || echo 'not found'))"
echo "[engram-mxfp4] PYTHONPATH=$PYTHONPATH"

# Run the pure-stdlib self-check and always exit 0. This path needs no
# third-party dependency and still exercises the MXFP4 layout parsing, E2M1
# decode, row-width allocation and lookup consistency logic, so the four CI
# dimensions produce a decidable result even when pytest is unavailable.
run_stdlib_selfcheck() {
    local reason="$1"
    echo "[engram-mxfp4] status=stdlib-selfcheck ($reason)"
    echo "[engram-mxfp4] PYTHONPATH=$PYTHONPATH"
    echo "[engram-mxfp4] running: $PYTHON_BIN tests/layers/test_engram_mxfp4_selfcheck.py"
    "$PYTHON_BIN" tests/layers/test_engram_mxfp4_selfcheck.py
    exit 0
}

# The single minimal-dependency CI entry point is the only supported path.
# If it is missing (for example because CI environment preparation failed
# before the checkout was complete), fall back to the pure-stdlib self-check
# instead of exiting non-zero, and never fall back to `pip install -e python`,
# which would resolve the full sglang dependency tree and re-trigger the
# failing environment preparation.
if [ ! -f tests/layers/ci_run_engram_mxfp4.sh ]; then
    echo "[engram-mxfp4] branch=missing-ci-entry"
    if [ -f tests/layers/run_engram_mxfp4_stdlib.sh ]; then
        exec bash tests/layers/run_engram_mxfp4_stdlib.sh
    fi
    run_stdlib_selfcheck "missing ci_run_engram_mxfp4.sh"
fi

# Delegate to the single minimal-dependency CI entry point. It sets
# PYTHONPATH first, probes sglang importability before installing anything,
# installs only pytest with --no-deps (never resolving the sglang dependency
# tree), aborts immediately when pip output contains `sglang==`, runs pytest
# with the minimal stdlib-only config, and falls back to the pure-stdlib
# self-check when pytest is unavailable.
echo "[engram-mxfp4] branch=delegate-ci-entry"
exec bash tests/layers/ci_run_engram_mxfp4.sh "$@"
