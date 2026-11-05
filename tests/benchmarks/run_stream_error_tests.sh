#!/usr/bin/env bash
# Run the benchmark stream-error tests in isolation.
#
# The repository root conftest imports the full sglang package, which drags in
# torch and other heavy runtime dependencies. These tests only need aiohttp,
# numpy and pytest, so the script installs the minimal requirements and runs
# pytest with the standalone config and --noconftest.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python -m pip install --disable-pip-version-check -r "${SCRIPT_DIR}/requirements-test.txt"

python -m pytest \
  -c "${SCRIPT_DIR}/pytest.ini" \
  --noconftest \
  "${SCRIPT_DIR}/test_stream_error_manifest.py" \
  "${SCRIPT_DIR}/test_stream_error_summary.py" \
  "$@"
