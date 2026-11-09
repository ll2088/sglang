#!/usr/bin/env bash
# Minimal environment preparation for the run_suite unit tests.
#
# Installs only test/unit/requirements.txt (pytest) and self-checks the
# interpreter, so an environment problem is reported as an environment
# failure instead of being mistaken for a test failure. Never resolves the
# full sglang dependency set.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQUIREMENTS="${SCRIPT_DIR}/requirements.txt"
PYTHON_BIN="${PYTHON:-python3}"

echo "[setup] using ${SCRIPT_DIR}/setup_env.sh"

for arg in "$@"; do
    case "${arg}" in
        -e|--editable|--editable=*|-e=*)
            echo "[setup] refusing editable install (${arg}); this script only" >&2
            echo "[setup] installs ${REQUIREMENTS} (pytest)." >&2
            exit 2
            ;;
    esac
done

if [ ! -f "${REQUIREMENTS}" ]; then
    echo "[setup] missing ${REQUIREMENTS}" >&2
    exit 2
fi

echo "[setup] installing ${REQUIREMENTS}"
"${PYTHON_BIN}" -m pip install --disable-pip-version-check -r "${REQUIREMENTS}"

echo "[setup] self-check: import pytest"
"${PYTHON_BIN}" -c "import pytest"

echo "[setup] self-check: pytest --version"
"${PYTHON_BIN}" -m pytest --version

echo "[setup] ready"
