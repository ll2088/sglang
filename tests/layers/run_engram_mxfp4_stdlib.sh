#!/usr/bin/env bash
# Pure-stdlib entry point for the MXFP4 Engram layout self-check.
#
# CI environment preparation can fail while resolving the full sglang
# dependency tree (torch, triton, py-spy, pybase64, pydantic, pyzmq,
# quack-kernels, regex, ...). The layout module
# python/sglang/srt/layers/engram_mxfp4.py is stdlib-only, so its checks must
# be runnable without installing the sglang package.
#
# This script installs nothing and never imports the sglang package. It runs
# the dependency-free self-check directly, so CI environment preparation
# cannot fail here. It is the single supported CI path for these tests and is
# invoked by run_engram_mxfp4_tests.sh and ci_install_minimal_deps.sh.
#
# Usage (from the repository root):
#   bash tests/layers/run_engram_mxfp4_stdlib.sh
#
# Exit code 0 means every self-check case passed.

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

echo "[engram-mxfp4] runner=run_engram_mxfp4_stdlib.sh"
echo "[engram-mxfp4] python=$PYTHON_BIN ($(command -v "$PYTHON_BIN"))"
echo "[engram-mxfp4] PYTHONPATH=$PYTHONPATH"
echo "[engram-mxfp4] selfcheck=$SELFCHECK"

if "$PYTHON_BIN" "$SELFCHECK"; then
    echo "[engram-mxfp4] status=passed"
    exit 0
fi

echo "[engram-mxfp4] status=failed" >&2
exit 1
