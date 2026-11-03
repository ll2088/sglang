#!/usr/bin/env bash
# Minimal, reproducible test environment for the registered quant tests.
# Deliberately avoids `pip install -e python` / `pip install -r requirements`
# because those resolve the full sglang dependency tree and are the root cause
# of the CI environment-preparation failures.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
. "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
python -m pip install \
  --no-deps \
  -c scripts/ci/constraints-ci.txt \
  -r scripts/ci/requirements-test-min.txt

# Install sglang itself without resolving its dependency tree. The tests that
# need torch/transformers are expected to be skipped or to rely on a prebuilt
# environment; the regression tests added here are stdlib-only.
if [ -f python/pyproject.toml ] || [ -f python/setup.py ]; then
  python -m pip install --no-deps -e python || true
fi

echo "--- environment ---"
python -V
python -m pip list
python -m pytest --version
