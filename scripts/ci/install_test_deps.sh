#!/usr/bin/env bash
# Install the minimal dependencies needed by the self-contained regression
# tests, then install the local sglang package without pulling its full
# runtime dependency tree. This keeps the test/stability/security dimensions
# independent from heavy runtime dependency resolution.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

python -m pip install --upgrade pip
python -m pip install -r "${SCRIPT_DIR}/requirements-test-min.txt" \
  -c "${SCRIPT_DIR}/constraints-ci.txt"

# Install the package itself without resolving runtime dependencies so that
# the self-contained tests can run even if a heavy dependency is unavailable.
if [ -d "${REPO_ROOT}/python" ]; then
  python -m pip install --no-deps -e "${REPO_ROOT}/python"
else
  python -m pip install --no-deps -e "${REPO_ROOT}"
fi
