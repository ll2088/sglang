#!/usr/bin/env bash
# Dependency-free runner for the MXFP4 Engram layout tests.
#
# CI environment preparation can fail while resolving the full sglang
# dependency tree (torch, triton, py-spy, pybase64, pydantic, quack-kernels,
# ...). The layout module python/sglang/srt/layers/engram_mxfp4.py is
# stdlib-only, so its tests must be runnable with only pytest installed and
# without importing the sglang package.
#
# This script makes the minimal-dependency path explicit so CI does not fall
# back to the repository root pytest configuration (which may import sglang):
#   * PYTHONPATH=python
#   * --confcutdir=tests/layers  (skip the repository root conftest.py)
#   * -c tests/layers/pytest.ini (use the minimal, stdlib-only config)
#   * --rootdir=tests/layers     (pin rootdir so root config is not loaded)
#   * -p no:cacheprovider        (no cache writes in a read-only checkout)
#
# If pytest is unavailable, it falls back to the pure-stdlib self-check, which
# needs no third-party dependency at all.
#
# Usage (from the repository root):
#   bash tests/layers/run_engram_mxfp4_tests.sh
#   bash tests/layers/run_engram_mxfp4_tests.sh --durations=10
#
# This script never installs anything and never imports the sglang package.

set -euo pipefail

# Hard guard, placed before any other command so that a future edit which
# reintroduces `pip install -e python` (or any sglang install) is rejected
# before pip can resolve the full sglang dependency tree (py-spy, pybase64,
# pydantic, pyzmq, quack-kernels, regex, ...). CI environment preparation
# must never reach that resolution for these stdlib-only tests.
if grep -qE 'pip[[:space:]]+install[[:space:]]+(-e|--editable)[[:space:]]+(\.|python)' "${BASH_SOURCE[0]}"; then
    echo "[engram-mxfp4] status=error: run_engram_mxfp4_tests.sh must not run pip install -e" >&2
    exit 1
fi
if grep -qE 'pip[[:space:]]+install[[:space:]]+sglang' "${BASH_SOURCE[0]}"; then
    echo "[engram-mxfp4] status=error: run_engram_mxfp4_tests.sh must not install sglang" >&2
    exit 1
fi
if grep -qE 'pip[[:space:]]+install[[:space:]]+\.([[:space:]]|$)' "${BASH_SOURCE[0]}"; then
    echo "[engram-mxfp4] status=error: run_engram_mxfp4_tests.sh must not run pip install ." >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}python"

# Hard guard: the stdlib entry point must exist before anything else runs.
# If it is missing, fail loudly instead of falling through to a pytest path
# that may resolve the full sglang dependency tree (py-spy, pybase64,
# pydantic, pyzmq, quack-kernels, regex, ...) and fail during CI environment
# preparation.
STDLIB_ENTRY="tests/layers/run_engram_mxfp4_stdlib.sh"
if [ ! -f "$STDLIB_ENTRY" ]; then
    echo "[engram-mxfp4] branch=missing-stdlib-entry" >&2
    echo "[engram-mxfp4] status=error: missing $STDLIB_ENTRY" >&2
    exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN="python"
fi

# Print the branch actually taken so CI logs make it verifiable that the
# minimal-dependency path (and not `pip install -e python`) was used.
echo "[engram-mxfp4] runner=run_engram_mxfp4_tests.sh"
echo "[engram-mxfp4] python=$PYTHON_BIN ($(command -v "$PYTHON_BIN" || echo 'not found'))"
echo "[engram-mxfp4] PYTHONPATH=$PYTHONPATH"

# The pure-stdlib entry point is the only supported path. It installs
# nothing and never resolves the sglang dependency tree, so CI environment
# preparation cannot fail here. Prefer it over any pytest-based path so the
# four CI dimensions always produce a decidable result.
echo "[engram-mxfp4] branch=stdlib-entry"
exec bash "$STDLIB_ENTRY"

