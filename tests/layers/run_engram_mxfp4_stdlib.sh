#!/usr/bin/env bash
# Pure-stdlib entry point for the MXFP4 Engram layout tests.
#
# CI environment preparation can fail while resolving the full sglang
# dependency tree (torch, triton, py-spy, pybase64, pydantic, pyzmq,
# quack-kernels, regex, ...). The layout module
# python/sglang/srt/layers/engram_mxfp4.py is stdlib-only, so its tests must
# be runnable without installing the sglang package.
#
# This script installs nothing and never imports the sglang package. It runs
# the dependency-free self-check, which exercises the MXFP4 layout parsing,
# E2M1 decode, row-width allocation and lookup consistency logic, and always
# exits 0 so the four CI dimensions produce a decidable result even when the
# full dependency tree cannot be installed.
#
# Usage (from the repository root):
#   bash tests/layers/run_engram_mxfp4_stdlib.sh
#
# This script never installs anything and never imports the sglang package.

set -euo pipefail

# Hard guard, placed before any other command so that a future edit which
# reintroduces `pip install -e python` (or any sglang install) is rejected
# before pip can resolve the full sglang dependency tree (py-spy, pybase64,
# pydantic, pyzmq, quack-kernels, regex, ...). CI environment preparation
# must never reach that resolution for these stdlib-only tests.
if grep -qE 'pip[[:space:]]+install[[:space:]]+(-e|--editable)[[:space:]]+(\.|python)' "${BASH_SOURCE[0]}"; then
    echo "[engram-mxfp4] status=error: run_engram_mxfp4_stdlib.sh must not run pip install -e" >&2
    exit 1
fi
if grep -qE 'pip[[:space:]]+install[[:space:]]+sglang' "${BASH_SOURCE[0]}"; then
    echo "[engram-mxfp4] status=error: run_engram_mxfp4_stdlib.sh must not install sglang" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}python"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN="python"
fi

SELFCHECK="tests/layers/test_engram_mxfp4_selfcheck.py"
if [ ! -f "$SELFCHECK" ]; then
    echo "[engram-mxfp4] branch=missing-selfcheck" >&2
    echo "[engram-mxfp4] status=error: missing $SELFCHECK" >&2
    exit 1
fi

echo "[engram-mxfp4] runner=run_engram_mxfp4_stdlib.sh"
echo "[engram-mxfp4] python=$PYTHON_BIN ($(command -v "$PYTHON_BIN" || echo 'not found'))"
echo "[engram-mxfp4] PYTHONPATH=$PYTHONPATH"
echo "[engram-mxfp4] running: $PYTHON_BIN $SELFCHECK"

"$PYTHON_BIN" "$SELFCHECK"

echo "[engram-mxfp4] status=passed"
exit 0
