#!/usr/bin/env bash
# Minimal environment preparation for the run_suite unit tests.
#
# Installs only test/unit/requirements.txt (pytest) and self-checks pytest so
# an environment problem is reported as an environment failure (exit 2)
# instead of being mistaken for a test failure. CI must call this script
# instead of `pip install -e python`, which resolves the full sglang
# dependency set (torch, kernels, ...) and can fail before any test runs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[setup] using ${SCRIPT_DIR}/setup_env.sh"

# 1) Reject editable installs first: they are the full-dependency path this
#    script exists to avoid.
for arg in "$@"; do
  case "${arg}" in
    -e|--editable|-e=*|--editable=*)
      echo "[setup] ERROR: editable installs are not supported; install test/unit/requirements.txt instead" >&2
      exit 2
      ;;
  esac
done

REQUIREMENTS="${SCRIPT_DIR}/requirements.txt"
if [[ ! -f "${REQUIREMENTS}" ]]; then
  echo "[setup] ERROR: missing requirements file: ${REQUIREMENTS}" >&2
  exit 2
fi

# 2) Install only the pytest-only requirements, capturing the log so the
#    full-dependency guard below can inspect it.
LOG_FILE="$(mktemp)"
trap 'rm -f "${LOG_FILE}"' EXIT

python -m pip install -r "${REQUIREMENTS}" 2>&1 | tee "${LOG_FILE}"

# 3) Fail fast if the full sglang dependency set was resolved anyway.
if grep -Eq 'Collecting .*from sglang==' "${LOG_FILE}"; then
  echo "[setup] ERROR: full sglang dependency set was resolved; the lightweight entry was bypassed" >&2
  exit 2
fi

# 4) Self-check pytest so a broken environment is distinguishable.
python -c 'import pytest'
python -m pytest --version

echo "[setup] pytest-only environment ready"
