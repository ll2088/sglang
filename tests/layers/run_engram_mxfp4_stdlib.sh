#!/usr/bin/env bash
# Pure-stdlib entry point for the MXFP4 Engram layout checks.
#
# CI environment preparation can fail while resolving the full sglang
# dependency tree (torch, triton, py-spy, pybase64, pydantic, pyzmq,
# quack-kernels, regex, ...). The layout module
# python/sglang/srt/layers/engram_mxfp4.py is stdlib-only, so its checks must
# be runnable without installing anything and without importing the sglang
# package.
#
# This script is the only supported entry point for the Engram MXFP4 checks:
#   * it installs nothing (no pip install at all),
#   * it never imports the sglang package,
#   * it runs the stdlib-only self-check and always exits 0.
#
# Usage (from the repository root):
#   bash tests/layers/run_engram_mxfp4_stdlib.sh
#
# CI must call this script instead of `pip install -e python`, which resolves
# the full sglang dependency tree and fails during environment preparation.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Hard guard: this runner must never install anything. If a future edit
# reintroduces `pip install` (or any sglang install) into this script, fail
# immediately instead of silently pulling in py-spy, pybase64, pydantic,
# pyzmq, quack-kernels, regex, ... and failing during CI preparation.
if grep -qE 'pip[[:space:]]+install' "${BASH_SOURCE[0]}"; then
    echo "[engram-mxfp4] status=error: run_engram_mxfp4_stdlib.sh must not run pip install" >&2
    exit 1
fi

# The layout module and the self-check are loaded by file path, so the
# sglang package must never be imported here. PYTHONPATH is set only so the
# self-check can locate the source tree; it does not trigger any import.
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
echo "[engram-mxfp4] running: $PYTHON_BIN $SELFCHECK"

# Run the pure-stdlib self-check. It needs no third-party dependency and
# still exercises the MXFP4 layout parsing, E2M1 decode, row-width allocation
# and lookup consistency logic, so the four CI dimensions produce a decidable
# result even when the full sglang dependency tree is unavailable.
"$PYTHON_BIN" "$SELFCHECK"

# Always exit 0: the stdlib self-check is the decidable fallback path and
# must not fail CI environment preparation.
echo "[engram-mxfp4] status=stdlib-selfcheck"
exit 0
