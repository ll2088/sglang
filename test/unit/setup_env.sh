#!/usr/bin/env bash
# Minimal environment preparation for the run_suite unit tests.
#
# These tests only exercise run_suite.py's exit-code and collection-diagnostic
# branches; conftest.py stubs out sglang.test.ci.* so the full sglang runtime
# (torch, kernels, ...) is not needed. This script installs only
# test/unit/requirements.txt (pytest) and self-checks the result, so an
# environment problem is reported as an environment failure instead of being
# mistaken for a test failure.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQUIREMENTS="${SCRIPT_DIR}/requirements.txt"

# Reject editable installs before any pip invocation: `pip install -e python`
# resolves the full sglang dependency set and can fail before pytest starts.
for arg in "$@"; do
    case "${arg}" in
        -e|--editable|--editable=*|-e=*)
            echo "[setup] error: editable installs are not supported here." >&2
            echo "[setup] install the pytest-only requirements instead:" >&2
            echo "[setup]   python -m pip install -r ${REQUIREMENTS}" >&2
            exit 2
            ;;
    esac
done

echo "[setup] using ${SCRIPT_DIR}/setup_env.sh"

if [ ! -f "${REQUIREMENTS}" ]; then
    echo "[setup] error: missing ${REQUIREMENTS}" >&2
    exit 2
fi

python -m pip install -r "${REQUIREMENTS}"

# Self-check so a broken environment is reported as an environment failure.
python -c "import pytest"
python -m pytest --version

echo "[setup] pytest-only environment ready"
