#!/usr/bin/env bash
# Install only the minimal dependencies required by the self-contained
# quant regression tests. This script deliberately does NOT install the
# sglang package itself (no `pip install -e python`), so that CI does not
# resolve the full runtime dependency tree.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

python -m pip install --upgrade pip
python -m pip install --no-deps -r "${REPO_ROOT}/requirements-test-min.txt"
