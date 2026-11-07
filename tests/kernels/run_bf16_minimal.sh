#!/usr/bin/env bash
# Run the BF16 capability tests with the minimal dependency set only.
# Never installs the sglang package, so py-spy/quack-kernels and friends are
# not pulled in during CI environment preparation.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${here}/../.." && pwd)"

python -m pip install --disable-pip-version-check -r "${here}/requirements-minimal.txt"

cd "${repo_root}"

# Stability: repeat via a shell loop instead of pytest-repeat, which is not in
# the minimal dependency set.
for i in 1 2 3; do
  echo "== bf16 capability run ${i}/3 =="
  python -m pytest -q -p no:randomly tests/kernels/test_dsa_topk_bf16_kernel.py
done
