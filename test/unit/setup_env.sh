#!/usr/bin/env bash
# Minimal environment preparation for the run_suite unit tests.
#
# These tests only exercise run_suite.py's exit-code and collection-diagnostic
# branches; test/unit/conftest.py stubs out sglang.test.ci.* so the full sglang
# runtime (torch, kernels, ...) is not needed. CI must call this script instead
# of `pip install -e python`, which resolves the full dependency set and can
# fail before any test runs.
#
# Usage (from the repository root):
#
#     bash test/unit/setup_env.sh
#     python test/unit/verify_run_suite.py
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[setup] using ${SCRIPT_DIR}/setup_env.sh"

# Reject editable installs explicitly: they resolve the full sglang dependency
# set and are exactly the failure mode this lightweight entry avoids. The guard
# must run before any pip invocation.
for arg in "$@"; do
  case "$arg" in
    -e|--editable|--editable=*|-e=*)
      echo "[setup] refusing editable install ($arg); use test/unit/requirements.txt" >&2
      exit 2
      ;;
  esac
done

PYTHON_BIN="${PYTHON:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[setup] python interpreter not found: $PYTHON_BIN" >&2
  exit 2
fi

REQUIREMENTS="${SCRIPT_DIR}/requirements.txt"
if [ ! -f "$REQUIREMENTS" ]; then
  echo "[setup] missing requirements file: $REQUIREMENTS" >&2
  exit 2
fi

# Install only the pytest-only requirements file. Never `pip install -e python`.
"$PYTHON_BIN" -m pip install --disable-pip-version-check -r "$REQUIREMENTS"

# Self-check so an environment problem is reported as an environment failure
# instead of being mistaken for a test failure.
"$PYTHON_BIN" -c "import pytest"
"$PYTHON_BIN" -m pytest --version

echo "[setup] ok: pytest-only environment ready"
