#!/usr/bin/env bash
# Minimal environment preparation for the run_suite unit tests.
#
# Installs only test/unit/requirements.txt (pytest) and self-checks the import
# so an environment problem is reported as an environment failure instead of
# being mistaken for a test failure. Never resolves the full sglang dependency
# set (`pip install -e python`).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQUIREMENTS="${SCRIPT_DIR}/requirements.txt"

[setup] using "${SCRIPT_DIR}/setup_env.sh"

# Reject editable installs before any pip invocation so the full sglang
# dependency resolution can never be triggered from this entry point.
for arg in "$@"; do
    case "${arg}" in
        -e|--editable|-e=*|--editable=*|-e./*|--editable./*)
            echo "[setup] error: editable installs are not supported here." >&2
            echo "[setup] install the pytest-only requirements with 'bash test/unit/setup_env.sh'." >&2
            exit 2
            ;;
    esac
done

if [ ! -f "${REQUIREMENTS}" ]; then
    echo "[setup] error: missing ${REQUIREMENTS}" >&2
    exit 2
fi

echo "[setup] pip install -r ${REQUIREMENTS}"
python -m pip install -r "${REQUIREMENTS}"

python -c 'import pytest'
python -m pytest --version

echo "[setup] ready: pytest installed, sglang runtime not required"
