#!/usr/bin/env bash
# Single authoritative environment setup for CI.
#
# Goals:
#   * Never inherit PYTHONPATH/PYTHONHOME from the runner.
#   * Rebuild .venv with an absolute python3.12, not whatever `python` is on PATH.
#   * Run every pip/pytest invocation through .venv/bin/python -m ... so the
#     interpreter that installs is the interpreter that tests.
#   * Fail loudly if pip resolves to the system interpreter.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

unset PYTHONPATH || true
unset PYTHONHOME || true
export PYTHONNOUSERSITE=1

PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3.12}"
if [ ! -x "$PYTHON_BIN" ]; then
  echo "error: $PYTHON_BIN not found; python3.12 is required" >&2
  exit 1
fi

"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'

"$PYTHON_BIN" -m venv --clear .venv

VENV_PY=".venv/bin/python"
if [ ! -x "$VENV_PY" ]; then
  echo "error: $VENV_PY missing after venv creation" >&2
  exit 1
fi

"$VENV_PY" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'

# Upgrade pip inside the venv before installing anything, so wheel tags are
# resolved for the venv interpreter rather than a stale bundled pip.
"$VENV_PY" -m pip install --upgrade pip

"$VENV_PY" -m pip install -e python

# Guard against installs landing in /usr/lib/python3/dist-packages.
"$VENV_PY" - <<'PY'
import sys
import pip

assert sys.version_info[:2] == (3, 12), sys.version
assert sys.prefix != sys.base_prefix, f"not in a venv: {sys.prefix}"
assert ".venv" in sys.executable, sys.executable
assert ".venv" in pip.__file__, pip.__file__
print("venv ok:", sys.executable)
print("pip ok:", pip.__file__)
PY

echo "setup_env.sh: environment ready"
