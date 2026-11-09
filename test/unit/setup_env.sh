#!/usr/bin/env bash
# Minimal environment preparation for the run_suite verification dimensions.
# Installs only test/unit/requirements.txt (pytest) and self-checks pytest so
# an environment problem is reported as an environment failure instead of
# being mistaken for a test failure.
#
# Do not run `pip install -e python` for these checks: it resolves the full
# sglang dependency set (torch, kernels, ...) and can fail before any test
# runs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQUIREMENTS="${SCRIPT_DIR}/requirements.txt"

echo "[setup] using ${SCRIPT_DIR}/setup_env.sh"
echo "[setup] requirements: ${REQUIREMENTS}"

# Reject editable installs before any pip invocation so the full sglang
# dependency resolution can never be triggered from this entry.
for arg in "$@"; do
    case "$arg" in
        -e|--editable|-e./*|-e=*|--editable=*)
            echo "[setup] error: editable installs are not supported here" >&2
            echo "[setup] use 'bash test/unit/setup_env.sh' (pytest only)" >&2
            exit 2
            ;;
    esac
done

if [ ! -f "${REQUIREMENTS}" ]; then
    echo "[setup] error: missing ${REQUIREMENTS}" >&2
    exit 2
fi

python -m pip install -r "${REQUIREMENTS}"

python -c "import pytest"
python -m pytest --version

echo "[setup] pytest-only environment ready"
