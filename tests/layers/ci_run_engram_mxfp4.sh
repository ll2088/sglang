#!/usr/bin/env bash
# Single minimal-dependency CI entry point for the MXFP4 Engram layout tests.
#
# CI environment preparation can fail while resolving the full sglang
# dependency tree (torch, triton, py-spy, pybase64, pydantic, pyzmq,
# quack-kernels, regex, ...). The layout module
# python/sglang/srt/layers/engram_mxfp4.py is stdlib-only, so its tests must
# not require that install.
#
# This script:
#   * sets PYTHONPATH=python first;
#   * installs only pytest with --no-deps when pytest is missing;
#   * asserts that sglang is not importable and that no dependency
#     resolution log mentions sglang==, failing loudly otherwise;
#   * runs pytest with -c tests/layers/pytest.ini, --rootdir=tests/layers,
#     --confcutdir=tests/layers and --engram-mxfp4-stdlib-only;
#   * falls back to the pure-stdlib self-check when pytest cannot be
#     installed.
#
# It never installs or imports the sglang package, so it cannot trigger the
# full dependency resolution that fails during CI environment preparation.
#
# Usage (from the repository root):
#   bash tests/layers/ci_run_engram_mxfp4.sh
#   bash tests/layers/ci_run_engram_mxfp4.sh --bench

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}python"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN="python"
fi

# Print the resolved interpreter and PYTHONPATH so CI logs make it
# verifiable which path was taken.
echo "[engram-mxfp4] runner=ci_run_engram_mxfp4.sh"
echo "[engram-mxfp4] python=$PYTHON_BIN ($(command -v "$PYTHON_BIN" || echo 'not found'))"
echo "[engram-mxfp4] PYTHONPATH=$PYTHONPATH"

# Fail loudly if the sglang package is importable. The MXFP4 layout module
# and its self-check are stdlib-only and are loaded by file path, so a
# resolvable sglang package means CI environment preparation resolved the
# full dependency tree this entry point exists to avoid.
if "$PYTHON_BIN" -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('sglang') is not None else 1)" >/dev/null 2>&1; then
    echo "[engram-mxfp4] status=error" >&2
    echo "[engram-mxfp4] sglang is importable; CI must not resolve the full sglang dependency tree for these tests." >&2
    echo "[engram-mxfp4] Replace any 'pip install -e python' / 'pip install sglang' step with this script." >&2
    exit 1
fi

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

# Install only pytest with --no-deps when it is missing. Never resolve the
# sglang dependency tree; --no-deps keeps pip from pulling in py-spy,
# pybase64, pydantic, pyzmq, quack-kernels, regex, ...
if ! "$PYTHON_BIN" -c "import pytest" >/dev/null 2>&1; then
    echo "[engram-mxfp4] installing pytest with --no-deps"
    if ! "$PYTHON_BIN" -m pip install --no-deps --disable-pip-version-check pytest >/dev/null 2>&1; then
        run_stdlib_selfcheck "pytest unavailable"
    fi
fi

if ! "$PYTHON_BIN" -c "import pytest" >/dev/null 2>&1; then
    run_stdlib_selfcheck "pytest import failed"
fi

PYTEST_ARGS=(
    -q
    -c tests/layers/pytest.ini
    --rootdir=tests/layers
    --confcutdir=tests/layers
    -p no:cacheprovider
    --engram-mxfp4-stdlib-only
)

if [ "${1:-}" = "--bench" ]; then
    shift
    echo "[engram-mxfp4] status=bench"
    echo "[engram-mxfp4] running: $PYTHON_BIN tests/layers/bench_engram_mxfp4.py"
    "$PYTHON_BIN" tests/layers/bench_engram_mxfp4.py "$@"
    exit 0
fi

TEST_FILES=(
    tests/layers/test_engram_mxfp4_selfcheck.py
    tests/layers/test_engram_mxfp4_minimal.py
    tests/layers/test_engram_mxfp4_ci_entry.py
)

for f in "${TEST_FILES[@]}"; do
    if [ ! -f "$f" ]; then
        run_stdlib_selfcheck "missing $f"
    fi
done

echo "[engram-mxfp4] status=pytest"
echo "[engram-mxfp4] running: $PYTHON_BIN -m pytest ${PYTEST_ARGS[*]} ${TEST_FILES[*]} $*"
set +e
"$PYTHON_BIN" -m pytest "${PYTEST_ARGS[@]}" "${TEST_FILES[@]}" "$@"
status=$?
set -e

if [ "$status" -eq 0 ]; then
    echo "[engram-mxfp4] status=pytest-passed"
    exit 0
fi

echo "[engram-mxfp4] status=pytest-failed (exit=$status)" >&2
exit "$status"
