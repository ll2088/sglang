#!/usr/bin/env bash
# Install the minimal dependency set required by the quantization regression
# tests.  This intentionally avoids resolving sglang's full runtime dependency
# tree (py-spy, pyzmq, quack-kernels, ...), which is slow and unnecessary for
# the tests under test/registered/quant/.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[install_test_deps] creating virtualenv at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip

# Install sglang without pulling its runtime dependencies, then install only
# the minimal test dependencies from the CI constraints file.
python -m pip install --no-deps -e python
python -m pip install -r constraints-ci.txt

echo "[install_test_deps] done; python=$(command -v python)"
python -c "import sys; print(sys.version)"
