#!/usr/bin/env bash
# Standalone runner for the DSA top-k BF16 capability tests.
#
# Deliberately avoids `pip install -e python` (which resolves the full sglang
# dependency set) and any CUDA/torch requirement: the tests load the pure-Python
# capability module by explicit file path.
#
# Usage:
#   tests/kernels/run_bf16_minimal.sh            # single run
#   tests/kernels/run_bf16_minimal.sh --count=3  # stability run
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

echo "python: $(command -v python || true)"
python --version
echo "pip: $(command -v python) -m pip"

# Install only the minimal test dependency, without resolving sglang's tree.
python -m pip install --no-deps -r tests/kernels/requirements-minimal.txt

EXTRA_ARGS=("$@")
python -m pytest -q -p no:randomly tests/kernels/test_dsa_topk_bf16.py "${EXTRA_ARGS[@]}"
