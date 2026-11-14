#!/usr/bin/env bash
# Minimal-dependency environment preparation for the MXFP4 Engram tests.
#
# CI environment preparation can fail while resolving the full sglang
# dependency tree (torch, triton, py-spy, pybase64, pydantic, quack-kernels,
# ...). The MXFP4 layout module python/sglang/srt/layers/engram_mxfp4.py is
# stdlib-only, so its tests only need pytest. This script installs pytest
# without resolving the sglang package, so CI preparation can exit 0 even when
# the full dependency tree is unavailable.
#
# It never installs the sglang package itself and never imports it.
#
# Usage (from the repository root):
#   bash tests/layers/ci_install_minimal_deps.sh
#
# After this script succeeds, run the tests with:
#   bash tests/layers/run_engram_mxfp4_tests.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Install only pytest, without resolving the sglang dependency tree.
# --no-deps keeps pip from pulling in py-spy, pybase64, pydantic,
# quack-kernels, regex, ... which is exactly the resolution that fails during
# CI environment preparation.
python -m pip install --no-deps --disable-pip-version-check pytest

# Sanity check: pytest must be importable and the layout module must load
# without importing sglang. This fails fast if the minimal environment is not
# actually usable, instead of failing later inside the test run.
python -c "import pytest"
PYTHONPATH=python python -c "
import importlib.util
spec = importlib.util.spec_from_file_location(
    'engram_mxfp4', 'python/sglang/srt/layers/engram_mxfp4.py'
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
layout = module.EngramLayout.from_metadata(
    256, (17, 19), {'engram_embed_quant': 'mxfp4'}
)
assert layout.row_bytes == 128
assert layout.total_row_bytes() == 136
print('minimal engram environment ready')
"
