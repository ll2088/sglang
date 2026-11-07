#!/usr/bin/env bash
# Run the BF16 indexer top-k capability tests in a minimal environment.
#
# The tests only need pytest and the standard library, so this script installs
# the minimal requirements and runs pytest directly on the test file instead of
# installing the sglang package (which would pull the full runtime dependency
# set and fail in a CPU-only CI job).
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${here}/../.." && pwd)"

python -m pip install --no-deps -r "${here}/requirements-minimal.txt"

cd "${repo_root}"
python -m pytest -q -p no:randomly "${here}/test_dsa_topk_bf16.py" "$@"
