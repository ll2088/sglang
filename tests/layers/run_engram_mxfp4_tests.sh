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

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}python"

SELFCHECK="tests/layers/test_engram_mxfp4_selfcheck.py"
MINIMAL="tests/layers/test_engram_mxfp4_minimal.py"
CI_ENTRY="tests/layers/test_engram_mxfp4_ci_entry.py"

if [ "${1:-}" = "--bench" ]; then
    shift
    exec python tests/layers/bench_engram_mxfp4.py "$@"
fi

if python -c "import pytest" >/dev/null 2>&1; then
    exec python -m pytest -q \
        -c tests/layers/pytest.ini \
        --rootdir=tests/layers \
        --confcutdir=tests/layers \
        -p no:cacheprovider \
        --engram-mxfp4-stdlib-only \
        "$@" \
        "$SELFCHECK" "$MINIMAL" "$CI_ENTRY"
fi

# pytest is not installed: delegate to the single minimal-dependency CI entry
# point, which installs only pytest --no-deps and otherwise falls back to the
# stdlib-only self-check. This keeps the runner usable in a minimal CI
# environment without resolving the full sglang dependency tree.
exec bash tests/layers/ci_run_engram_mxfp4.sh "$@"

# pytest is not installed: run the stdlib-only self-check directly.
# This path never imports sglang/torch/triton, so it stays runnable when the
# full sglang dependency tree cannot be installed during CI preparation.
echo "pytest not available; running stdlib-only self-check" >&2
exec python "$SELFCHECK"
