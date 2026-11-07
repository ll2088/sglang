#!/usr/bin/env bash
# Run the BF16 top-k capability tests in a minimal environment.
#
# The tests are pure Python and must not trigger sglang's full dependency
# resolution.  Install only the minimal requirements with --no-deps and run
# pytest against the explicit test path.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

python -m pip install --no-deps -r tests/kernels/requirements-minimal.txt

python -m pytest -q tests/kernels/test_dsa_topk_bf16.py "$@"
