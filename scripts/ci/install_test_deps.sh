#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Minimal dependency installer for the sonar-fix regression tests.
# Installs this repo without resolving its full runtime dependency tree,
# then installs only the packages the tests actually need.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

echo "== python: $(python --version 2>&1)"
echo "== pip: $(python -m pip --version 2>&1)"

python -m pip install --upgrade pip

# Install the project itself without pulling in runtime deps.
python -m pip install --no-deps -e .

# Install the minimal test requirements only.
python -m pip install -r "${REPO_ROOT}/requirements-test-min.txt"

echo "== installed packages (filtered):"
python -m pip list | grep -Ei 'pytest|sglang' || true

echo "== install_test_deps.sh done"
