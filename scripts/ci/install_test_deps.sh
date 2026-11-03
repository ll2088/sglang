#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Install the minimal dependency set required by the self-contained regression
# tests under test/registered/quant/. This script intentionally avoids
# `pip install -e python` so CI does not pull the full sglang runtime stack.
#
# Usage: scripts/ci/install_test_deps.sh
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "error: ${PYTHON_BIN} not found" >&2
    exit 1
fi

echo "python: $(${PYTHON_BIN} -V 2>&1)"
echo "pip:    $(${PYTHON_BIN} -m pip -V 2>&1)"

# Pin pytest to a version that supports the Python interpreter in use.
# The regression tests only need stdlib + pytest.
if ! "${PYTHON_BIN}" -m pip install --disable-pip-version-check \
        -c "$(dirname "$0")/constraints-ci.txt" \
        "pytest>=7.4,<9"; then
    echo "error: failed to install pytest" >&2
    "${PYTHON_BIN}" -m pip -V >&2 || true
    exit 1
fi

echo "test dependencies installed"
