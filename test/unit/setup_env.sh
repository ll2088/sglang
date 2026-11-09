#!/usr/bin/env bash
# Minimal environment preparation for the run_suite unit tests.
#
# These tests only exercise run_suite.py's exit-code and collection-diagnostic
# branches; conftest.py stubs out sglang.test.ci.* so the full sglang runtime
# (torch, kernels, ...) is not needed. CI must call this script instead of
# `pip install -e python`, which resolves the full dependency set and can fail
# before any test runs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[setup] using ${SCRIPT_DIR}/setup_env.sh"

# Guard: never let an editable install of the full package slip in here.
for arg in "$@"; do
  case "${arg}" in
    -e|--editable|*/python)
      echo "[setup] refusing full sglang install argument: ${arg}" >&2
      exit 2
      ;;
  esac
done

PYTHON_BIN="${PYTHON:-python3}"

"${PYTHON_BIN}" -m pip install --disable-pip-version-check \
  -r "${SCRIPT_DIR}/requirements.txt"

# Self-check so an environment problem is reported as such instead of being
# mistaken for a test failure.
"${PYTHON_BIN}" -c "import pytest; print('[setup] pytest', pytest.__version__)"
"${PYTHON_BIN}" -m pytest --version

echo "[setup] ok"
