#!/usr/bin/env bash
# Install the minimal dependency set needed to run the quantization
# regression tests in CI. sglang itself is installed with --no-deps so
# that pip does not try to resolve the full runtime dependency tree.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install -c constraints-ci.txt -r /dev/stdin <<'EOF'
pytest
pytest-randomly
numpy
torch
EOF

# Install sglang without pulling its runtime dependencies.
"${PYTHON_BIN}" -m pip install --no-deps -e python

# Fail fast if the package cannot be imported.
"${PYTHON_BIN}" -c "import sglang"
"${PYTHON_BIN}" -m pip check
