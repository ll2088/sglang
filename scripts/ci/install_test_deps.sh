#!/usr/bin/env bash
# Install the minimal dependency set for the quant regression tests.
#
# The tests under test/registered/quant/ are intentionally self-contained and
# do not import sglang, so we must NOT run `pip install -e python` (which
# resolves the full sglang dependency tree). Instead we install the local
# package without dependencies and add only the minimal test requirements.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install \
  -r "${SCRIPT_DIR}/requirements-test-min.txt" \
  -c "${SCRIPT_DIR}/constraints-ci.txt"

# Install the local package without pulling in its runtime dependencies.
if [ -d "${REPO_ROOT}/python" ]; then
  "${PYTHON_BIN}" -m pip install --no-deps -e "${REPO_ROOT}/python"
fi

"${PYTHON_BIN}" -m pytest --version
