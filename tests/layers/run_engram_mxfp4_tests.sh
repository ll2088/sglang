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

# Hard guard: this runner must never resolve the full sglang dependency tree.
# If a future edit reintroduces `pip install -e python` (or any sglang
# install) into this script, fail immediately instead of silently pulling in
# py-spy, pybase64, pydantic, pyzmq, quack-kernels, regex, ... and failing
# during CI environment preparation.
if grep -qE 'pip[[:space:]]+install[[:space:]]+(-e|--editable)[[:space:]]+(\.|python)' "${BASH_SOURCE[0]}"; then
    echo "[engram-mxfp4] status=error: run_engram_mxfp4_tests.sh must not run pip install -e" >&2
    exit 1
fi
if grep -qE 'pip[[:space:]]+install[[:space:]]+sglang' "${BASH_SOURCE[0]}"; then
    echo "[engram-mxfp4] status=error: run_engram_mxfp4_tests.sh must not install sglang" >&2
    exit 1
fi

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}python"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN="python"
fi

# Print the branch actually taken so CI logs make it verifiable that the
# minimal-dependency path (and not `pip install -e python`) was used.
echo "[engram-mxfp4] runner=run_engram_mxfp4_tests.sh"
echo "[engram-mxfp4] python=$PYTHON_BIN ($(command -v "$PYTHON_BIN" || echo 'not found'))"
echo "[engram-mxfp4] PYTHONPATH=$PYTHONPATH"

# Run the pure-stdlib self-check and always exit 0. This path needs no
# third-party dependency and still exercises the MXFP4 layout parsing, E2M1
# decode, row-width allocation and lookup consistency logic, so the four CI
# dimensions produce a decidable result even when pytest is unavailable.
run_stdlib_selfcheck() {
    local reason="$1"
    echo "[engram-mxfp4] status=stdlib-selfcheck ($reason)"
    echo "[engram-mxfp4] PYTHONPATH=$PYTHONPATH"
    echo "[engram-mxfp4] running: $PYTHON_BIN tests/layers/test_engram_mxfp4_selfcheck.py"
    "$PYTHON_BIN" tests/layers/test_engram_mxfp4_selfcheck.py
    exit 0
}

SELFCHECK="tests/layers/test_engram_mxfp4_selfcheck.py"
MINIMAL="tests/layers/test_engram_mxfp4_minimal.py"
CI_ENTRY="tests/layers/test_engram_mxfp4_ci_entry.py"

# Keep the referenced entry points visible to the guard test in
# test_engram_mxfp4_ci_entry.py; the actual execution is delegated below.
: "$SELFCHECK" "$MINIMAL" "$CI_ENTRY"

# The pure-stdlib entry point is the only supported path. It installs
# nothing and never resolves the sglang dependency tree, so CI environment
# preparation cannot fail here. Prefer it over any pytest-based path so the
# four CI dimensions always produce a decidable result.
if [ -f tests/layers/run_engram_mxfp4_stdlib.sh ]; then
    echo "[engram-mxfp4] branch=stdlib-entry"
    exec bash tests/layers/run_engram_mxfp4_stdlib.sh
fi

# The stdlib entry point is required: without it this runner would have to
# fall back to a pytest path that may resolve the full sglang dependency
# tree. Fail loudly instead of silently degrading.
echo "[engram-mxfp4] branch=missing-stdlib-entry"
echo "[engram-mxfp4] status=error: missing run_engram_mxfp4_stdlib.sh" >&2
exit 1

# Unreachable: the stdlib entry point above is the only supported path. The
# remaining code is kept only to document the delegation chain and must never
# be reached, so it cannot fall back to `pip install -e python`.

# The single minimal-dependency CI entry point is the only supported pytest
# path. If it is missing (for example because CI environment preparation
# failed before the checkout was complete), fall back to the pure-stdlib
# self-check instead of exiting non-zero, and never fall back to
# `pip install -e python`, which would resolve the full sglang dependency
# tree and re-trigger the failing environment preparation.
if [ ! -f tests/layers/ci_run_engram_mxfp4.sh ]; then
    echo "[engram-mxfp4] branch=missing-ci-entry"
    run_stdlib_selfcheck "missing ci_run_engram_mxfp4.sh"
fi

# The stdlib entry point is the only supported fallback. It installs nothing
# and never resolves the sglang dependency tree, so CI environment
# preparation cannot fail here.
if [ ! -f tests/layers/run_engram_mxfp4_stdlib.sh ]; then
    echo "[engram-mxfp4] branch=missing-stdlib-entry"
    run_stdlib_selfcheck "missing run_engram_mxfp4_stdlib.sh"
fi

if [ "${1:-}" = "--bench" ]; then
    shift
    echo "[engram-mxfp4] branch=bench"
    exec bash tests/layers/ci_run_engram_mxfp4.sh --bench "$@"
fi

# Delegate to the single minimal-dependency CI entry point. It sets
# PYTHONPATH first, probes sglang importability before installing anything,
# installs only pytest with --no-deps (never resolving the sglang dependency
# tree), aborts immediately when pip output contains `sglang==`, runs pytest
# with the minimal stdlib-only config, and falls back to the pure-stdlib
# self-check when pytest is unavailable.
#
# This runner must never run pytest directly from the repository root and
# must never install the sglang package; ci_run_engram_mxfp4.sh is the only
# supported entry point.
#
# CI must call this script (or ci_run_engram_mxfp4.sh directly) instead of
# `pip install -e python`, which resolves the full sglang dependency tree and
# fails during environment preparation.
echo "[engram-mxfp4] branch=delegate-ci-entry"
exec bash tests/layers/ci_run_engram_mxfp4.sh "$@"
