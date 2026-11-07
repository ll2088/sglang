#!/usr/bin/env bash
# Run the BF16 top-k capability tests without installing the sglang meta-package.
# Usage: tests/kernels/run_bf16_minimal.sh [extra pytest args...]
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${here}/../.." && pwd)"

python -m pip install --disable-pip-version-check -r "${here}/requirements-minimal.txt"

# Make the repository importable without installing it.
export PYTHONPATH="${repo_root}/python${PYTHONPATH:+:${PYTHONPATH}}"

cd "${here}"
python -m pytest -q -c pytest.ini test_dsa_topk_bf16_kernel.py "$@"
