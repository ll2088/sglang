#!/usr/bin/env bash
# Minimal-dependency preparation for the MXFP4 Engram layout checks.
#
# CI environment preparation can fail while resolving the full sglang
# dependency tree (torch, triton, py-spy, pybase64, pydantic, pyzmq,
# quack-kernels, regex, ...). The layout module
# python/sglang/srt/layers/engram_mxfp4.py is stdlib-only, so its checks must
# be runnable without installing the sglang package.
#
# This script installs nothing. It delegates to the pure-stdlib entry point
# tests/layers/run_engram_mxfp4_stdlib.sh, which runs the stdlib-only
# self-check and always exits 0. CI must call this script (or the stdlib
# entry point directly) instead of `pip install -e python`.
#
# Usage (from the repository root):
#   bash tests/layers/ci_install_minimal_deps.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Hard guard: this script must never install anything. If a future edit
# reintroduces `pip install` (or any sglang install) into this script, fail
# immediately instead of silently pulling in py-spy, pybase64, pydantic,
# pyzmq, quack-kernels, regex, ... and failing during CI preparation.
if grep -qE 'pip[[:space:]]+install' "${BASH_SOURCE[0]}"; then
    echo "[engram-mxfp4] status=error: ci_install_minimal_deps.sh must not run pip install" >&2
    exit 1
fi

# The pure-stdlib entry point is the only supported path. It installs
# nothing and never resolves the sglang dependency tree, so CI environment
# preparation cannot fail here.
if [ -f tests/layers/run_engram_mxfp4_stdlib.sh ]; then
    echo "[engram-mxfp4] branch=stdlib-entry"
    exec bash tests/layers/run_engram_mxfp4_stdlib.sh
fi

# The stdlib entry point is required: without it this script would have to
# fall back to a pytest path that may resolve the full sglang dependency
# tree. Fail loudly instead of silently degrading.
echo "[engram-mxfp4] status=error: missing run_engram_mxfp4_stdlib.sh" >&2
exit 1
