#!/usr/bin/env bash
# Run the multimodal benchmark stream-error tests without installing the
# full GPU dependency tree. Intended for CPU-only CI runners.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

python -m pip install --upgrade pip
python -m pip install -r "${SCRIPT_DIR}/requirements-test.txt"

cd "${REPO_ROOT}"
python -m pytest -q \
  "python/sglang/multimodal_gen/benchmarks/test_stream_error_event.py" \
  "$@"
