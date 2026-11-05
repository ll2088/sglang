#!/usr/bin/env bash
# Run the stream-error benchmark tests without installing the full sglang
# GPU dependency tree. Intended for CPU-only CI jobs.
set -euxo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

python -m pip install --upgrade pip
python -m pip install -r tests/benchmarks/requirements-test.txt

python -m pytest -q \
  tests/benchmarks/test_stream_error_detection.py \
  tests/benchmarks/test_stream_error_summary.py
