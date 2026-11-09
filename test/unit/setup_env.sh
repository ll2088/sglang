#!/usr/bin/env bash
# Minimal environment preparation for the run_suite unit tests.
#
# Installs only test/unit/requirements.txt (pytest) and self-checks pytest so
# an environment problem is reported as an environment failure instead of
# being mistaken for a test failure. It must never resolve the full sglang
# dependency set (torch, kernels, py-spy, quack-kernels, ...).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQUIREMENTS="${SCRIPT_DIR}/requirements.txt"
LOG_FILE="$(mktemp)"
trap 'rm -f "${LOG_FILE}"' EXIT

# Reject editable installs: they resolve the full sglang dependency set.
for arg in "$@"; do
    case "${arg}" in
        -e|--editable|-e*|--editable=*)
            echo "[setup] ERROR: editable installs are not supported; " \
                 "install ${REQUIREMENTS} instead" >&2
            exit 2
            ;;
    esac
done

if [[ ! -f "${REQUIREMENTS}" ]]; then
    echo "[setup] ERROR: missing ${REQUIREMENTS}" >&2
    exit 2
fi

echo "[setup] using ${REQUIREMENTS}"

# Install the pytest-only requirements, capturing the log so the smoke guard
# below can assert the full sglang dependency set was not resolved.
python -m pip install --disable-pip-version-check -r "${REQUIREMENTS}" \
    2>&1 | tee "${LOG_FILE}"

# Smoke guard: fail fast if the install log shows sglang dependency resolution.
if grep -Eq 'Collecting .*from sglang==' "${LOG_FILE}"; then
    echo "[setup] ERROR: full sglang dependency set was resolved; " \
         "the lightweight entry was not used" >&2
    exit 2
fi

# Self-check pytest so a broken environment is distinguishable from a test
# failure.
python -c 'import pytest'
python -m pytest --version

echo "[setup] pytest ready"
