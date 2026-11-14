#!/usr/bin/env bash
# Minimal-dependency preparation for the MXFP4 Engram layout tests.
#
# CI environment preparation can fail while resolving the full sglang
# dependency tree (torch, triton, py-spy, pybase64, pydantic, pyzmq,
# quack-kernels, regex, ...). The layout module
# python/sglang/srt/layers/engram_mxfp4.py is stdlib-only, so its tests must
# be runnable without installing the sglang package.
#
# This script installs nothing. It delegates to the pure-stdlib entry point
# tests/layers/run_engram_mxfp4_stdlib.sh, which needs no third-party
# dependency at all. If that entry point is missing, it fails loudly instead
# of falling back to `pip install -e python` (which would resolve the full
# sglang dependency tree and re-trigger the failing environment preparation).
#
# Usage (from the repository root):
#   bash tests/layers/ci_install_minimal_deps.sh
#
# This script never installs anything and never imports the sglang package.

set -euo pipefail

# Hard guard, placed before any other command so that a future edit which
# reintroduces `pip install -e python` (or any sglang install) is rejected
# before pip can resolve the full sglang dependency tree (py-spy, pybase64,
# pydantic, pyzmq, quack-kernels, regex, ...). CI environment preparation
# must never reach that resolution for these stdlib-only tests.
if grep -qE 'pip[[:space:]]+install[[:space:]]+(-e|--editable)[[:space:]]+(\.|python)' "${BASH_SOURCE[0]}"; then
    echo "[engram-mxfp4] status=error: ci_install_minimal_deps.sh must not run pip install -e" >&2
    exit 1
fi
if grep -qE 'pip[[:space:]]+install[[:space:]]+sglang' "${BASH_SOURCE[0]}"; then
    echo "[engram-mxfp4] status=error: ci_install_minimal_deps.sh must not install sglang" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}python"

STDLIB_ENTRY="tests/layers/run_engram_mxfp4_stdlib.sh"

# The pure-stdlib entry point is the only supported path. It installs
# nothing and never resolves the sglang dependency tree, so CI environment
# preparation cannot fail here.
if [ -f "$STDLIB_ENTRY" ]; then
    echo "[engram-mxfp4] branch=stdlib-entry"
    echo "[engram-mxfp4] PYTHONPATH=$PYTHONPATH"
    exec bash "$STDLIB_ENTRY"
fi

# The stdlib entry point is required: without it this script would have to
# fall back to a pytest path that may resolve the full sglang dependency
# tree. Fail loudly instead of silently degrading.
echo "[engram-mxfp4] branch=missing-stdlib-entry" >&2
echo "[engram-mxfp4] status=error: missing $STDLIB_ENTRY" >&2
exit 1
