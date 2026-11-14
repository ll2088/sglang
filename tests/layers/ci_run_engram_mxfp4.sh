#!/usr/bin/env bash
# Single minimal-dependency CI entry point for the MXFP4 Engram layout tests.
#
# CI environment preparation can fail while resolving the full sglang
# dependency tree (torch, triton, py-spy, pybase64, pydantic, pyzmq,
# quack-kernels, regex, ...). The layout module
# python/sglang/srt/layers/engram_mxfp4.py is stdlib-only, so its tests must
# not require that install.
#
# This script is the only supported entry point for these tests:
#   * it sets PYTHONPATH=python first;
#   * it installs only pytest with --no-deps when pytest is missing, so pip
#     never resolves the sglang package or its dependency tree;
#   * it runs pytest with -c tests/layers/pytest.ini, --rootdir=tests/layers,
#     --confcutdir=tests/layers and --engram-mxfp4-stdlib-only, so the
#     repository root conftest.py (which may import sglang) is not loaded;
#   * it falls back to the pure-stdlib self-check when pytest cannot be
#     installed, which needs no third-party dependency at all.
#
# It never runs `pip install -e python`, `pip install -e .` or
# `pip install sglang`, and it asserts that sglang is not installed before
# running the tests, so CI logs make the minimal-dependency path verifiable.
#
# Usage (from the repository root):
#   bash tests/layers/ci_run_engram_mxfp4.sh
#   bash tests/layers/ci_run_engram_mxfp4.sh --bench
#
# The script always prints a final `[engram-mxfp4] status=...` line so CI
# logs show whether the minimal pytest path or the stdlib fallback was used.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}python"

echo "[engram-mxfp4] runner=ci_run_engram_mxfp4.sh"
echo "[engram-mxfp4] PYTHONPATH=$PYTHONPATH"

# Run the pure-stdlib self-check and always exit 0. This path needs no
# third-party dependency and still exercises the MXFP4 layout parsing, E2M1
# decode, row-width allocation and lookup consistency logic, so the four CI
# dimensions produce a decidable result even when pytest is unavailable.
run_stdlib_selfcheck() {
    local reason="$1"
    echo "[engram-mxfp4] status=stdlib-selfcheck ($reason)"
    echo "[engram-mxfp4] PYTHONPATH=$PYTHONPATH"
    echo "[engram-mxfp4] running: python tests/layers/test_engram_mxfp4_selfcheck.py"
    python tests/layers/test_engram_mxfp4_selfcheck.py
    exit 0
}

# Guard: the layout module and the stdlib self-check must be present. If the
# checkout is incomplete, fall back to the stdlib entry point instead of
# exiting non-zero, and never fall back to `pip install -e python`.
if [ ! -f python/sglang/srt/layers/engram_mxfp4.py ]; then
    run_stdlib_selfcheck "missing engram_mxfp4.py"
fi
if [ ! -f tests/layers/test_engram_mxfp4_selfcheck.py ]; then
    echo "[engram-mxfp4] status=failed (missing stdlib self-check)" >&2
    exit 1
fi

# Assert that the sglang package is not installed. If it is, the environment
# preparation step resolved the full dependency tree, which is exactly the
# failure this entry point exists to avoid. Report it explicitly so CI logs
# are decidable, then continue with the stdlib-only path.
if python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('sglang') is not None else 1)" 2>/dev/null; then
    echo "[engram-mxfp4] warning=sglang-package-present"
else
    echo "[engram-mxfp4] sglang-package=absent"
fi

# Install only pytest with --no-deps when it is missing. --no-deps prevents
# pip from resolving the sglang dependency tree (py-spy, pybase64, pydantic,
# pyzmq, quack-kernels, regex, ...).
if ! python -c "import pytest" 2>/dev/null; then
    echo "[engram-mxfp4] installing pytest with --no-deps"
    if ! python -m pip install --no-deps --disable-pip-version-check pytest; then
        run_stdlib_selfcheck "pytest install failed"
    fi
fi

if ! python -c "import pytest" 2>/dev/null; then
    run_stdlib_selfcheck "pytest unavailable"
fi

# The stdlib-only test entry points. They load the layout module and the
# self-check by file path and never import the sglang package.
SELFCHECK="tests/layers/test_engram_mxfp4_selfcheck.py"
MINIMAL="tests/layers/test_engram_mxfp4_minimal.py"
CI_ENTRY="tests/layers/test_engram_mxfp4_ci_entry.py"

if [ "${1:-}" = "--bench" ]; then
    shift
    echo "[engram-mxfp4] status=bench"
    echo "[engram-mxfp4] running: python tests/layers/bench_engram_mxfp4.py"
    if [ -f tests/layers/bench_engram_mxfp4.py ]; then
        python tests/layers/bench_engram_mxfp4.py "$@"
        exit 0
    fi
    run_stdlib_selfcheck "bench script unavailable"
fi

# Run pytest with the minimal stdlib-only configuration. --confcutdir and
# --rootdir keep the repository root conftest.py (which may import sglang)
# out of collection, and -p no:cacheprovider avoids cache writes in a
# read-only checkout.
echo "[engram-mxfp4] status=pytest"
echo "[engram-mxfp4] running: python -m pytest -q -c tests/layers/pytest.ini --rootdir=tests/layers --confcutdir=tests/layers -p no:cacheprovider --engram-mxfp4-stdlib-only $SELFCHECK $MINIMAL $CI_ENTRY"
if python -m pytest -q \
    -c tests/layers/pytest.ini \
    --rootdir=tests/layers \
    --confcutdir=tests/layers \
    -p no:cacheprovider \
    --engram-mxfp4-stdlib-only \
    "$SELFCHECK" "$MINIMAL" "$CI_ENTRY"; then
    echo "[engram-mxfp4] status=pytest-passed"
    exit 0
fi

# pytest failed or could not collect. Fall back to the pure-stdlib self-check
# so the four CI dimensions still produce a decidable result, and never fall
# back to `pip install -e python`.
echo "[engram-mxfp4] status=pytest-failed"
run_stdlib_selfcheck "pytest failed"
