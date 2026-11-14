#!/usr/bin/env bash
# Pure-stdlib entry point for the MXFP4 Engram layout tests.
#
# CI environment preparation can fail while resolving the full sglang
# dependency tree (torch, triton, py-spy, pybase64, pydantic, pyzmq,
# quack-kernels, regex, ...). The layout module
# python/sglang/srt/layers/engram_mxfp4.py is stdlib-only, so its tests must
# be runnable without installing the sglang package.
#
# This script installs nothing, imports no third-party dependency and never
# imports the sglang package. It runs the stdlib-only self-check by file
# path, so CI environment preparation cannot fail here.
#
# Usage (from the repository root):
#   bash tests/layers/run_engram_mxfp4_stdlib.sh
#
# It always prints a final `[engram-mxfp4] status=...` line so CI logs show
# which path was taken.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}python"

SELFCHECK="tests/layers/test_engram_mxfp4_selfcheck.py"
if [ ! -f "$SELFCHECK" ]; then
    echo "[engram-mxfp4] status=error: missing $SELFCHECK" >&2
    exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN="python"
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "[engram-mxfp4] status=error: no python interpreter found" >&2
    exit 1
fi

echo "[engram-mxfp4] entry=run_engram_mxfp4_stdlib.sh"
echo "[engram-mxfp4] python=$PYTHON_BIN ($(command -v "$PYTHON_BIN"))"
echo "[engram-mxfp4] PYTHONPATH=$PYTHONPATH"

if "$PYTHON_BIN" "$SELFCHECK"; then
    echo "[engram-mxfp4] status=passed"
    exit 0
fi

echo "[engram-mxfp4] status=failed" >&2
exit 1
