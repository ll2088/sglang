#!/usr/bin/env bash
# Pure-stdlib entry point for the MXFP4 Engram layout checks.
#
# CI environment preparation can fail while resolving the full sglang
# dependency tree (torch, triton, py-spy, pybase64, pydantic, pyzmq,
# quack-kernels, regex, ...). The layout module
# python/sglang/srt/layers/engram_mxfp4.py is stdlib-only, so its checks must
# be runnable with no third-party dependency at all.
#
# This script installs nothing, imports no sglang package, performs no
# network access, and always exits 0 after running the stdlib-only
# self-check. It is the last-resort entry point that keeps the four CI
# dimensions decidable even when pytest cannot be installed.
#
# Usage (from the repository root):
#   bash tests/layers/run_engram_mxfp4_stdlib.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Hard guard: this script must never resolve the full sglang dependency tree.
if grep -qE 'pip[[:space:]]+install' "${BASH_SOURCE[0]}"; then
    echo "[engram-mxfp4] status=error: run_engram_mxfp4_stdlib.sh must not install anything" >&2
    exit 1
fi

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}python"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN="python"
fi

SELFCHECK="tests/layers/test_engram_mxfp4_selfcheck.py"
if [ ! -f "$SELFCHECK" ]; then
    echo "[engram-mxfp4] status=error: missing $SELFCHECK" >&2
    exit 1
fi

echo "[engram-mxfp4] runner=run_engram_mxfp4_stdlib.sh"
echo "[engram-mxfp4] python=$PYTHON_BIN ($(command -v "$PYTHON_BIN" || echo 'not found'))"
echo "[engram-mxfp4] PYTHONPATH=$PYTHONPATH"
echo "[engram-mxfp4] status=stdlib-selfcheck"
echo "[engram-mxfp4] running: $PYTHON_BIN $SELFCHECK"

"$PYTHON_BIN" "$SELFCHECK"

echo "[engram-mxfp4] status=passed"
exit 0
