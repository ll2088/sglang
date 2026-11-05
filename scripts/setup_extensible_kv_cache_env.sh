#!/usr/bin/env bash
# Prepare an isolated Python 3.12 environment for the extensible KV cache
# checks. The script is intentionally strict: any failure aborts so that the
# caller never falls back to the system interpreter.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"

# Drop inherited interpreter overrides so the venv is not bypassed.
unset PYTHONPATH || true
unset PYTHONHOME || true

# Pick an explicit Python 3.12 interpreter; never rely on a bare `python`.
PYTHON_BIN=""
for candidate in /usr/bin/python3.12 /usr/local/bin/python3.12 python3.12; do
    if command -v "${candidate}" >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v "${candidate}")"
        break
    fi
done
if [[ -z "${PYTHON_BIN}" ]]; then
    echo "error: python3.12 not found; install it before running this script" >&2
    exit 1
fi

# Recreate the venv from scratch so a stale environment cannot leak in.
rm -rf "${VENV_DIR}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"

VENV_PYTHON="${VENV_DIR}/bin/python"
if [[ ! -x "${VENV_PYTHON}" ]]; then
    echo "error: venv python missing at ${VENV_PYTHON}" >&2
    exit 1
fi

# Fail fast if the venv is not actually Python 3.12.
"${VENV_PYTHON}" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'

# Always go through the venv interpreter; never call bare pip/python.
"${VENV_PYTHON}" -m pip install --upgrade pip
"${VENV_PYTHON}" -m pip install -e "${REPO_ROOT}/python"

# Diagnostics to make interpreter mix-ups obvious in CI logs.
echo "--- environment diagnostics ---"
echo "which python: $(command -v python || true)"
echo "python -V: $("${VENV_PYTHON}" -V)"
echo "pip -V: $("${VENV_PYTHON}" -m pip -V)"
"${VENV_PYTHON}" -c 'import sys; print("sys.path:", sys.path)'

# Smoke import so a broken install fails here rather than in pytest.
"${VENV_PYTHON}" -c 'import sglang'

echo "extensible KV cache environment ready at ${VENV_DIR}"
