#!/usr/bin/env bash
# Prepare a Python 3.12 virtualenv for the extensible KV cache checks.
#
# The verification commands all start with `. .venv/bin/activate`, so this
# script must create that venv with python3.12 and install the project with
# the venv's own pip. Using the system pip resolves cp314 wheels and fails.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "error: $PYTHON_BIN not found; install Python 3.12 first" >&2
  exit 1
fi

"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'

rm -rf .venv
"$PYTHON_BIN" -m venv .venv
# shellcheck disable=SC1091
. .venv/bin/activate

VENV_PY="$REPO_ROOT/.venv/bin/python"
"$VENV_PY" -V
"$VENV_PY" -m pip install --upgrade pip setuptools wheel
"$VENV_PY" -m pip -V

# Show the tags pip will accept so a cp314 resolution is visible in the log.
"$VENV_PY" -m pip debug --verbose | sed -n '1,40p'

# Refuse wheels built for another interpreter (e.g. cp314) and require the
# venv interpreter to be used for the editable install.
PIP_LOG="$(mktemp)"
if ! "$VENV_PY" -m pip install \
    --only-binary=:all: \
    --python-version 3.12 \
    --implementation cp \
    --abi cp312 \
    -e python 2>&1 | tee "$PIP_LOG"; then
  echo "error: pip install failed; full log follows" >&2
  cat "$PIP_LOG" >&2
  rm -f "$PIP_LOG"
  exit 1
fi
rm -f "$PIP_LOG"

"$VENV_PY" -V
"$VENV_PY" -m pip -V
"$VENV_PY" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
