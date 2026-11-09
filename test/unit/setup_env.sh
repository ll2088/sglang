#!/usr/bin/env bash
# Lightweight environment preparation for the run_suite verification jobs.
#
# These checks only exercise run_suite.py's exit-code and collection-diagnostic
# branches; test/unit/conftest.py stubs out sglang.test.ci.* so the full sglang
# runtime (torch, kernels, ...) is not needed. Installing `-e python` resolves
# the full dependency set and can fail before any test runs, so this script
# installs only test/unit/requirements.txt (pytest).
#
# Usage (from anywhere):
#
#     bash test/unit/setup_env.sh
#
# The resolved script path and arguments are printed so CI logs prove this
# lightweight entry was hit instead of a full `pip install -e python`.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQUIREMENTS="${SCRIPT_DIR}/requirements.txt"
LOG_FILE="${SCRIPT_DIR}/.setup_env.log"

echo "[setup] using ${SCRIPT_DIR}/setup_env.sh"
echo "[setup] args: $*"
echo "[setup] requirements: ${REQUIREMENTS}"

# Reject editable installs before any pip invocation: an editable sglang
# install resolves the full dependency set and is exactly what previously
# failed during environment preparation.
for arg in "$@"; do
    case "${arg}" in
        -e|--editable|-e=*|--editable=*)
            echo "[setup] ERROR: editable installs are not supported here;" >&2
            echo "[setup] use 'bash test/unit/setup_env.sh' (pytest only)." >&2
            exit 2
            ;;
    esac
done

if [ ! -f "${REQUIREMENTS}" ]; then
    echo "[setup] ERROR: missing ${REQUIREMENTS}" >&2
    exit 2
fi

python -m pip install -r "${REQUIREMENTS}" 2>&1 | tee "${LOG_FILE}"

# Guard: the install log must not show the full sglang dependency set being
# resolved. If it does, the lightweight entry was bypassed.
if grep -Eq 'Collecting .*from sglang==' "${LOG_FILE}"; then
    echo "[setup] ERROR: full sglang dependency set was resolved;" >&2
    echo "[setup] the lightweight entry was bypassed." >&2
    exit 2
fi

python -c 'import pytest'
python -m pytest --version

echo "[setup] done"
