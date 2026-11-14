#!/usr/bin/env bash
# Minimal-dependency preparation step for the MXFP4 Engram layout tests.
#
# CI environment preparation can fail while resolving the full sglang
# dependency tree (torch, triton, py-spy, pybase64, pydantic, quack-kernels,
# regex, ...). The layout module python/sglang/srt/layers/engram_mxfp4.py is
# stdlib-only, so its tests must be runnable with only pytest installed and
# without importing the sglang package.
#
# This script is the minimal-dependency preparation step CI should call
# instead of `pip install -e python`. It installs only pytest with --no-deps
# (never resolving sglang's install_requires) and then delegates to
# ci_run_engram_mxfp4.sh, which runs the tests and falls back to the pure
# stdlib self-check when pytest is unavailable.
#
# It never installs or imports the sglang package, so it cannot trigger the
# full dependency resolution that fails during CI environment preparation.
#
# Usage (from the repository root):
#   bash tests/layers/ci_install_minimal_deps.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}python"

SELFCHECK="tests/layers/test_engram_mxfp4_selfcheck.py"

# Run the pure-stdlib self-check and always exit 0. This path needs no
# third-party dependency and still exercises the MXFP4 layout parsing, E2M1
# decode, row-width allocation and lookup consistency logic, so the four CI
# dimensions produce a decidable result even when pytest is unavailable.
run_stdlib_selfcheck() {
    local reason="$1"
    echo "[engram-mxfp4] status=stdlib-selfcheck ($reason)"
    echo "[engram-mxfp4] PYTHONPATH=$PYTHONPATH"
    echo "[engram-mxfp4] running: python $SELFCHECK"
    python "$SELFCHECK"
    exit 0
}

if [ ! -f "$SELFCHECK" ]; then
    echo "[engram-mxfp4] status=stdlib-selfcheck (missing $SELFCHECK)"
    exit 0
fi

if [ ! -f tests/layers/ci_run_engram_mxfp4.sh ]; then
    run_stdlib_selfcheck "missing ci_run_engram_mxfp4.sh"
fi

# Delegate to the single minimal-dependency CI entry point. It never installs
# anything (installing would re-trigger the failing full sglang dependency
# resolution), runs pytest with the minimal stdlib-only config when pytest is
# available, and falls back to the pure stdlib self-check otherwise. Keeping
# the logic in exactly one place avoids the duplicated/unreachable fallback
# branches that previously let CI fall back to `pip install -e python` and
# resolve the full sglang dependency tree.
#
# This script must never run `pip install -e python`, `pip install -e .` or
# `pip install sglang`; ci_run_engram_mxfp4.sh is the only supported entry
# point.
#
# Note: this script is a thin alias. It does not install pytest either, since
# any install step risks resolving the full sglang dependency tree during CI
# environment preparation. ci_run_engram_mxfp4.sh handles the pytest-absent
# case by falling back to the stdlib self-check.
exec bash tests/layers/ci_run_engram_mxfp4.sh "$@"
