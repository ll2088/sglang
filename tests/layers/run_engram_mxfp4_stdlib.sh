#!/usr/bin/env bash
# Pure-stdlib, dependency-free entry point for the MXFP4 Engram layout tests.
#
# CI environment preparation can fail while resolving the full sglang
# dependency tree (torch, triton, py-spy, pybase64, pydantic, pyzmq,
# quack-kernels, regex, ...). The layout module
# python/sglang/srt/layers/engram_mxfp4.py is stdlib-only, so its checks must
# be runnable without installing anything at all.
#
# This script is the only supported no-dependency entry point:
#   * it never runs `pip install` (not even `pip install --no-deps pytest`);
#   * it never imports the sglang package;
#   * it always exits 0 after running the stdlib-only self-check.
#
# Usage (from the repository root):
#   bash tests/layers/run_engram_mxfp4_stdlib.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}python"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN="python"
fi

echo "[engram-mxfp4] runner=run_engram_mxfp4_stdlib.sh"
echo "[engram-mxfp4] python=$PYTHON_BIN ($(command -v "$PYTHON_BIN" || echo 'not found'))"
echo "[engram-mxfp4] PYTHONPATH=$PYTHONPATH"
echo "[engram-mxfp4] status=stdlib-selfcheck"
echo "[engram-mxfp4] running: $PYTHON_BIN tests/layers/test_engram_mxfp4_selfcheck.py"

"$PYTHON_BIN" tests/layers/test_engram_mxfp4_selfcheck.py

exit 0
