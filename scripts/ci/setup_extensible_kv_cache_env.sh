#!/usr/bin/env bash
# Prepare the environment for the extensible KV cache tests.
#
# The verification plan requires Python 3.12. Relying on the ambient
# interpreter can resolve to a newer Python (e.g. 3.14) and pull cp314
# wheels, which is what broke the previous CI run. This script therefore
# creates a dedicated .venv with an explicit 3.12 interpreter and installs
# the project into it.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3.12}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "error: $PYTHON_BIN not found; install Python 3.12 or set PYTHON_BIN" >&2
  exit 1
fi

# Fail loudly if the interpreter is not the required version.
"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info[:2] != (3, 12):
    raise SystemExit(
        f"expected Python 3.12, got {sys.version.split()[0]} "
        f"({sys.executable})"
    )
PY

if [ ! -d .venv ]; then
  "$PYTHON_BIN" -m venv .venv
fi

# shellcheck disable=SC1091
. .venv/bin/activate

# Print the interpreter actually in use so a wrong pin is obvious in CI logs.
python -V
python -c 'import sys; print("executable:", sys.executable)'
pip -V

# Install with the venv interpreter explicitly; never fall back to the
# system pip. Keep the full output visible on failure.
python -m pip install --upgrade pip
python -m pip install -e python

echo "extensible KV cache environment ready"
