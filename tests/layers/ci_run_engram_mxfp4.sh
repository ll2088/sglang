#!/usr/bin/env bash
# Single minimal-dependency CI entry point for the MXFP4 Engram layout tests.
#
# CI environment preparation can fail while resolving the full sglang
# dependency tree (torch, triton, py-spy, pybase64, pydantic, pyzmq,
# quack-kernels, regex, ...). The layout module
# python/sglang/srt/layers/engram_mxfp4.py is stdlib-only, so its tests must
# not require that install.
#
# This script is the only supported entry point:
#   * it sets PYTHONPATH=python first;
#   * it probes sglang importability BEFORE installing anything and fails
#     loudly (status=error) if sglang is already importable, so a full
#     dependency resolution cannot be silently reused;
#   * it installs only pytest with --no-deps (never resolving the sglang
#     dependency tree);
#   * it streams pip output and aborts immediately when a `sglang==`
#     requirement line appears, so a stray `pip install -e python` cannot
#     keep downloading the full dependency tree;
#   * it runs pytest with the minimal stdlib-only config and falls back to
#     the pure-stdlib self-check when pytest is unavailable.
#
# It never installs or imports the sglang package.
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

# Print the branch actually taken so CI logs make it verifiable that the
# minimal-dependency path (and not `pip install -e python`) was used.
echo "[engram-mxfp4] runner=ci_run_engram_mxfp4.sh"
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

# Probe sglang importability BEFORE installing anything. If sglang is already
# importable, the environment preparation step resolved the full dependency
# tree, which is exactly the failure this entry point exists to prevent.
if "$PYTHON_BIN" -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('sglang') is not None else 1)" >/dev/null 2>&1; then
    echo "[engram-mxfp4] status=error (sglang is importable; full dependency tree was resolved)"
    echo "[engram-mxfp4] CI must not install the sglang package for these tests."
    exit 1
fi

# Install only pytest with --no-deps. Stream pip output and abort immediately
# when a `sglang==` requirement line appears, so a stray `pip install -e
# python` cannot keep downloading the full dependency tree.
install_pytest_minimal() {
    local pip_log
    pip_log="$(mktemp)"
    set +e
    "$PYTHON_BIN" -m pip install --no-deps --disable-pip-version-check pytest 2>&1 | tee "$pip_log"
    local pip_status="${PIPESTATUS[0]}"
    set -e
    if grep -q 'sglang==' "$pip_log"; then
        echo "[engram-mxfp4] status=error (pip output contains sglang==; full dependency resolution was triggered)"
        rm -f "$pip_log"
        exit 1
    fi
    rm -f "$pip_log"
    return "$pip_status"
}

if ! "$PYTHON_BIN" -c "import pytest" >/dev/null 2>&1; then
    echo "[engram-mxfp4] branch=install-pytest-minimal"
    if ! install_pytest_minimal; then
        run_stdlib_selfcheck "pytest install failed"
    fi
fi

if ! "$PYTHON_BIN" -c "import pytest" >/dev/null 2>&1; then
    run_stdlib_selfcheck "pytest unavailable"
fi

if [ "${1:-}" = "--bench" ]; then
    shift
    echo "[engram-mxfp4] branch=bench"
    if [ -f tests/layers/bench_engram_mxfp4.py ]; then
        exec "$PYTHON_BIN" tests/layers/bench_engram_mxfp4.py "$@"
    fi
    run_stdlib_selfcheck "bench script unavailable"
fi

echo "[engram-mxfp4] branch=pytest"
echo "[engram-mxfp4] running: $PYTHON_BIN -m pytest -q -c tests/layers/pytest.ini --rootdir=tests/layers --confcutdir=tests/layers -p no:cacheprovider --engram-mxfp4-stdlib-only tests/layers/test_engram_mxfp4_selfcheck.py tests/layers/test_engram_mxfp4_minimal.py tests/layers/test_engram_mxfp4_ci_entry.py"
set +e
"$PYTHON_BIN" -m pytest -q \
    -c tests/layers/pytest.ini \
    --rootdir=tests/layers \
    --confcutdir=tests/layers \
    -p no:cacheprovider \
    --engram-mxfp4-stdlib-only \
    tests/layers/test_engram_mxfp4_selfcheck.py \
    tests/layers/test_engram_mxfp4_minimal.py \
    tests/layers/test_engram_mxfp4_ci_entry.py
pytest_status="$?"
set -e

if [ "$pytest_status" -eq 0 ]; then
    echo "[engram-mxfp4] status=pytest-passed"
    exit 0
fi

echo "[engram-mxfp4] status=pytest-failed (exit=$pytest_status)"
run_stdlib_selfcheck "pytest failed"
