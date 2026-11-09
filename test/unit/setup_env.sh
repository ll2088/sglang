#!/usr/bin/env bash
# Minimal environment preparation for the run_suite unit tests.
#
# Installs only test/unit/requirements.txt (pytest). The unit tests stub out
# sglang.test.ci.* in conftest.py, so the full sglang runtime (torch, kernels,
# ...) is not needed. Never run `pip install -e python` here: it resolves the
# full dependency set and can fail before pytest starts.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQUIREMENTS="${SCRIPT_DIR}/requirements.txt"

echo "[setup] using ${SCRIPT_DIR}/setup_env.sh"

# Guard: an editable install would pull the full sglang dependency set.
for arg in "$@"; do
  case "${arg}" in
    -e|--editable|-e=*|--editable=*)
      echo "[setup] refusing editable install (${arg}); only pytest is required" >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "${REQUIREMENTS}" ]]; then
  echo "[setup] missing requirements file: ${REQUIREMENTS}" >&2
  exit 2
fi

python -m pip install -r "${REQUIREMENTS}"

# Self-check so an environment problem is reported as an environment failure
# instead of being mistaken for a test failure.
python -c "import pytest"
python -m pytest --version

echo "[setup] pytest ready"
