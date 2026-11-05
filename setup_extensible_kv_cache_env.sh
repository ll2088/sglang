#!/usr/bin/env bash
# Prepare an isolated Python 3.12 environment for the extensible KV cache
# tests. The venv must be bound to python3.12 explicitly: a bare `python3 -m
# venv` can pick up a newer interpreter and then pip resolves cp314 wheels,
# which fails against the 3.12 target.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3.12}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "ERROR: $PYTHON_BIN not found on PATH" >&2
    echo "which python3.12: $(command -v python3.12 || echo '<missing>')" >&2
    echo "python3 -V: $(python3 -V 2>&1 || echo '<missing>')" >&2
    exit 1
fi

# Recreate the venv so a stale interpreter cannot leak in from a previous run.
rm -rf .venv
"$PYTHON_BIN" -m venv .venv

# shellcheck disable=SC1091
source .venv/bin/activate

# Fail fast if the venv is not actually bound to 3.12.
VENV_VERSION="$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
if [ "$VENV_VERSION" != "3.12" ]; then
    echo "ERROR: .venv python is $VENV_VERSION, expected 3.12" >&2
    echo "which python3.12: $(command -v python3.12 || echo '<missing>')" >&2
    echo "python -V: $(python -V 2>&1)" >&2
    exit 1
fi

# Keep pip inside the venv: no user site, no system dist-packages, no
# inherited PYTHONPATH pointing at the host interpreter.
unset PYTHONPATH || true
export PIP_REQUIRE_VIRTUALENV=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

python -m pip install --upgrade --isolated pip setuptools wheel

# Install the project in editable mode. --isolated keeps pip from reading
# user/site config that could redirect to system packages.
python -m pip install --isolated -e .

# Diagnostics for the next CI run.
echo "--- environment diagnostics ---"
python -c 'import sys; print(sys.executable, sys.version)'
python -m pip -V
python -m pip list | head -n 40
