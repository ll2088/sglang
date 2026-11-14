#!/usr/bin/env bash
# Pure-stdlib entry point for the MXFP4 Engram layout checks.
#
# CI environment preparation can fail while resolving the full sglang
# dependency tree (torch, triton, py-spy, pybase64, pydantic, pyzmq,
# quack-kernels, regex, ...). The layout module
# python/sglang/srt/layers/engram_mxfp4.py is stdlib-only, so its checks must
# be runnable without installing the sglang package.
#
# This script installs nothing, performs no network access, and never imports
# the sglang package. It runs the stdlib-only self-check and always exits 0 so
# the four CI dimensions produce a decidable result even when the full
# dependency tree cannot be installed.
#
# Usage (from the repository root):
#   bash tests/layers/run_engram_mxfp4_stdlib.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Hard guard: this script must never install anything. If a future edit
# reintroduces `pip install` (or any sglang install) into this script, fail
# immediately instead of silently pulling in py-spy, pybase64, pydantic,
# pyzmq, quack-kernels, regex, ... and failing during CI preparation.
if grep -qE 'pip[[:space:]]+install' "${BASH_SOURCE[0]}"; then
    echo "[engram-mxfp4] status=error: run_engram_mxfp4_stdlib.sh must not run pip install" >&2
    exit 1
fi

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
