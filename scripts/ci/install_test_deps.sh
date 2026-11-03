#!/usr/bin/env bash
# Install the minimal dependency set required by the quantization regression
# tests. The sglang package itself is installed with --no-deps so that pip
# does not resolve the full runtime dependency tree (py-spy, pyzmq, ...).
set -euxo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install -r constraints-ci.txt

# Install sglang without pulling its runtime dependencies. The editable
# install only needs the package metadata to be importable.
"${PYTHON_BIN}" -m pip install --no-deps -e python

# Fail fast if the package cannot be imported at all.
"${PYTHON_BIN}" -c "import sglang; print('sglang import ok')"
