#!/usr/bin/env bash
# Pure-stdlib entry point for the MXFP4 Engram layout checks.
#
# CI environment preparation can fail while resolving the full sglang
# dependency tree (torch, triton, py-spy, pybase64, pydantic, quack-kernels,
# regex, ...). The layout module python/sglang/srt/layers/engram_mxfp4.py is
# stdlib-only, so its checks must be runnable without installing anything.
#
# This script never runs `pip install` (in particular it never runs
# `pip install -e python` / `pip install -e .` / `pip install sglang`), never
# imports the sglang package, and always exits 0 after running the
# dependency-free self-check. It is the fallback used by
# run_engram_mxfp4_tests.sh and ci_install_minimal_deps.sh when the single
# minimal-dependency CI entry point is unavailable.
#
# Usage (from the repository root):
#   bash tests/layers/run_engram_mxfp4_stdlib.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}python"

echo "[engram-mxfp4] status=stdlib-selfcheck"
echo "[engram-mxfp4] PYTHONPATH=$PYTHONPATH"
echo "[engram-mxfp4] running: python tests/layers/test_engram_mxfp4_selfcheck.py"
python tests/layers/test_engram_mxfp4_selfcheck.py
exit 0
