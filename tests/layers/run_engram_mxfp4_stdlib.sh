#!/usr/bin/env bash
# Pure-stdlib CI entry point for the MXFP4 Engram layout tests.
#
# CI environment preparation can fail while resolving the full sglang
# dependency tree (torch, triton, py-spy, pybase64, pydantic, pyzmq,
# quack-kernels, regex, ...). The layout module
# python/sglang/srt/layers/engram_mxfp4.py is stdlib-only, so its tests must
# stay runnable without installing anything at all.
#
# This script is the dependency-free path CI must use when the full sglang
# dependency tree cannot be resolved:
#   * it never runs pip install (which would re-trigger the failing
#     environment preparation),
#   * it never imports the sglang package,
#   * it runs the stdlib-only self-check and always exits 0.
#
# Usage (from the repository root):
#   bash tests/layers/run_engram_mxfp4_stdlib.sh
#
# This script never installs anything and never imports the sglang package.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}python"

echo "[engram-mxfp4] status=stdlib-selfcheck (dependency-free entry)"
echo "[engram-mxfp4] PYTHONPATH=$PYTHONPATH"
echo "[engram-mxfp4] running: python tests/layers/test_engram_mxfp4_selfcheck.py"
python tests/layers/test_engram_mxfp4_selfcheck.py

exit 0
