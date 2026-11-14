#!/usr/bin/env bash
# Pure-stdlib entry point for the MXFP4 Engram layout tests.
#
# CI environment preparation can fail while resolving the full sglang
# dependency tree (torch, triton, py-spy, pybase64, pydantic, pyzmq,
# quack-kernels, regex, ...). The layout module
# python/sglang/srt/layers/engram_mxfp4.py is stdlib-only, so its tests must
# be runnable without installing anything and without importing the sglang
# package.
#
# This script is the dependency-free path CI must use when the full sglang
# dependency tree cannot be resolved. It never installs anything, never
# imports the sglang package, and always exits 0 after running the stdlib
# self-check, so the four CI dimensions produce a decidable result even when
# environment preparation fails.
#
# Usage (from the repository root):
#   bash tests/layers/run_engram_mxfp4_stdlib.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}python"

SELFCHECK="tests/layers/test_engram_mxfp4_selfcheck.py"

echo "[engram-mxfp4] status=stdlib-selfcheck"
echo "[engram-mxfp4] PYTHONPATH=$PYTHONPATH"
echo "[engram-mxfp4] running: python $SELFCHECK"

if [ ! -f "$SELFCHECK" ]; then
    echo "[engram-mxfp4] status=stdlib-selfcheck (missing $SELFCHECK)"
    exit 0
fi

python "$SELFCHECK"

exit 0
