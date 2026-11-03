#!/usr/bin/env bash
# Install the minimal dependency set for the quant regression tests.
#
# The tests under test/registered/quant are self-contained (stdlib + pytest)
# and must not trigger a full `pip install -e python` dependency resolution.
# Installing sglang with --no-deps keeps the environment small and avoids
# pulling py-spy / pybase64 / pydantic / pyzmq / quack-kernels / regex etc.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"

set -x

"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install \
    -r "${SCRIPT_DIR}/requirements-test-min.txt" \
    -c "${SCRIPT_DIR}/constraints-ci.txt"

# Install the local package without resolving its runtime dependencies.
# --no-build-isolation avoids pulling build backends that are not needed
# for the pure-Python test modules.
"${PYTHON_BIN}" -m pip install \
    --no-deps \
    --no-build-isolation \
    -e "${REPO_ROOT}/python"

set +x
