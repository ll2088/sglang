#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Install the minimal dependency set needed by the registered quant tests
# without letting pip resolve the full sglang dependency tree.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

CONSTRAINTS="${REPO_ROOT}/scripts/ci/constraints-ci.txt"
REQUIREMENTS="${REPO_ROOT}/scripts/ci/requirements-test-min.txt"

if [[ ! -f "${CONSTRAINTS}" ]]; then
  echo "missing constraints file: ${CONSTRAINTS}" >&2
  exit 1
fi
if [[ ! -f "${REQUIREMENTS}" ]]; then
  echo "missing requirements file: ${REQUIREMENTS}" >&2
  exit 1
fi

python -m pip install --upgrade pip

# Install the local package without dependency resolution. This is the key
# step: a plain `pip install -e python` would pull py-spy, pybase64,
# quack-kernels and the rest of the sglang runtime dependency set.
python -m pip install --no-deps -e "${REPO_ROOT}/python"

# Install only the minimal test dependencies, constrained by the CI pins.
python -m pip install \
  -c "${CONSTRAINTS}" \
  -r "${REQUIREMENTS}"

echo "test dependencies installed (no full sglang dependency resolution)"
