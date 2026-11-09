#!/usr/bin/env bash
# Minimal environment preparation for the run_suite unit tests.
#
# These tests only exercise run_suite.py's exit-code and collection-diagnostic
# branches; conftest.py stubs out sglang.test.ci.* so the full sglang runtime
# (torch, kernels, ...) is not needed. CI must call this script instead of
# `pip install -e python`, which resolves the full dependency set and can fail
# during environment preparation before any test runs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQUIREMENTS="${SCRIPT_DIR}/requirements.txt"

# Resolve the interpreter explicitly so CI logs prove which python is used.
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    PYTHON_BIN="python"
fi

echo "[setup] using ${SCRIPT_DIR}/setup_env.sh"
echo "[setup] python: $("${PYTHON_BIN}" --version 2>&1)"
echo "[setup] pip: $("${PYTHON_BIN}" -m pip --version 2>&1)"

# Guard: refuse to run when the caller is trying to install the full sglang
# package in editable mode. This keeps the lightweight path from silently
# degrading into the full dependency resolution that broke CI.
for arg in "$@"; do
    case "${arg}" in
        --editable|--editable=*|-e|-e=*)
            echo "[setup] ERROR: editable installs are not supported here;" >&2
            echo "[setup] use ${REQUIREMENTS} (pytest only) instead." >&2
            exit 2
            ;;
    esac
done

if [ ! -f "${REQUIREMENTS}" ]; then
    echo "[setup] ERROR: missing ${REQUIREMENTS}" >&2
    exit 2
fi

# Install only the pytest-only requirements file. Never `pip install -e`.
"${PYTHON_BIN}" -m pip install --disable-pip-version-check -r "${REQUIREMENTS}"

# Self-check so an environment problem is reported as an environment failure
# instead of being mistaken for a test failure.
"${PYTHON_BIN}" - <<'PY'
import sys

try:
    import pytest
except ImportError as exc:  # pragma: no cover - environment guard
    print(f"[setup] ERROR: pytest import failed: {exc}", file=sys.stderr)
    sys.exit(2)

print(f"[setup] pytest {pytest.__version__} import OK")
PY

"${PYTHON_BIN}" -m pytest --version

echo "[setup] done"
