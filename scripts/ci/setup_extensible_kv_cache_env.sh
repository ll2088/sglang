#!/usr/bin/env bash
# Prepare an isolated Python 3.12 environment for the extensible KV cache
# checks. Every pip/pytest invocation in CI must go through this script so the
# system interpreter and its dist-packages are never used.
set -euo pipefail

# Never inherit interpreter search paths from the caller.
unset PYTHONPATH || true
unset PYTHONHOME || true

if ! command -v python3.12 >/dev/null 2>&1; then
    echo "python3.12 not found on PATH" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"

if [ ! -x "${VENV_DIR}/bin/python" ]; then
    python3.12 -m venv "${VENV_DIR}"
fi

PY="${VENV_DIR}/bin/python"

"${PY}" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'

"${PY}" -m pip install --upgrade pip
# Only accept prebuilt wheels so pip never resolves a cp314 wheel for a
# different interpreter, and never falls back to a source build.
"${PY}" -m pip install --only-binary=:all: -e "${REPO_ROOT}/python"

"${PY}" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
"${PY}" - <<'PY'
import sys

print("sys.executable:", sys.executable)
for entry in sys.path:
    print("sys.path:", entry)
    assert "/usr/lib/python3/dist-packages" not in entry, entry

import pydantic  # noqa: F401
import pybase64  # noqa: F401
import sglang  # noqa: F401
PY

echo "extensible KV cache environment ready: ${PY}"
