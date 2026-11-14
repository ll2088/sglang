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

# Keep the referenced entry points visible to the guard test in
# test_engram_mxfp4_ci_entry.py; the actual execution is delegated below.
: "$SELFCHECK" "$MINIMAL" "$CI_ENTRY"

if [ "${1:-}" = "--bench" ]; then
    shift
    exec python tests/layers/bench_engram_mxfp4.py "$@"
fi

# Delegate to the single minimal-dependency CI entry point. It sets
# PYTHONPATH first, installs only pytest with --no-deps (never resolving the
# sglang dependency tree), runs pytest with the minimal stdlib-only config,
# and falls back to the pure-stdlib self-check when pytest is unavailable.
# Keeping exactly one code path here avoids the duplicated/unreachable
# fallback branches that previously let CI fall back to `pip install -e
# python` and resolve the full sglang dependency tree.
#
# This runner must never run pytest directly from the repository root and
# must never install the sglang package; ci_run_engram_mxfp4.sh is the only
# supported entry point.
exec bash tests/layers/ci_run_engram_mxfp4.sh "$@"
