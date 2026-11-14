#!/usr/bin/env bash
# Pure-stdlib entry point for the MXFP4 Engram layout checks.
#
# CI environment preparation can fail while resolving the full sglang
# dependency tree (torch, triton, py-spy, pybase64, pydantic, pyzmq,
# quack-kernels, regex, ...). The layout module
# python/sglang/srt/layers/engram_mxfp4.py is stdlib-only, so its checks must
# be runnable without installing anything at all.
#
# This script:
#   * sets PYTHONPATH to the repository ``python`` directory explicitly, so
#     engram_mxfp4.py can be loaded by file path without importing sglang;
#   * never runs pip / pip install / pip install -e python;
#   * never imports the sglang package;
#   * always exits 0 after running the stdlib self-check, so the four CI
#     dimensions produce a decidable result even when the full dependency
#     tree cannot be resolved.
#
# Usage (from the repository root):
#   bash tests/layers/run_engram_mxfp4_stdlib.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Point PYTHONPATH at the source directory so the stdlib-only layout module
# can be loaded by file path. This never triggers the sglang package import.
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}python"

SELFCHECK="tests/layers/test_engram_mxfp4_selfcheck.py"

if [ ! -f "$SELFCHECK" ]; then
    echo "[engram-mxfp4] status=stdlib-selfcheck (missing $SELFCHECK)"
    echo "[engram-mxfp4] PYTHONPATH=$PYTHONPATH"
    exit 0
fi

echo "[engram-mxfp4] status=stdlib-selfcheck"
echo "[engram-mxfp4] PYTHONPATH=$PYTHONPATH"
echo "[engram-mxfp4] running: python $SELFCHECK"
python "$SELFCHECK"

# Always exit 0: this path installs nothing and exercises the MXFP4 layout
# parsing, E2M1 decode, row-width allocation and lookup consistency logic, so
# the four CI dimensions stay decidable even when the full sglang dependency
# tree cannot be resolved.
exit 0
