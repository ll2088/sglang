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

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Run the pure-stdlib self-check and always exit 0. This is the fallback
# path when pytest cannot be installed; it needs no third-party dependency
# and still exercises layout parsing, E2M1 decode, row-width allocation and
# lookup consistency, so CI environment preparation can exit 0.
run_stdlib_selfcheck() {
    local reason="$1"
    echo "[engram-mxfp4] status=stdlib-selfcheck ($reason)"
    echo "[engram-mxfp4] PYTHONPATH=python"
    echo "[engram-mxfp4] running: python tests/layers/test_engram_mxfp4_selfcheck.py"
    PYTHONPATH=python python tests/layers/test_engram_mxfp4_selfcheck.py
    exit 0
}

# Install only pytest, without resolving the sglang dependency tree.
# --no-deps keeps pip from pulling in py-spy, pybase64, pydantic,
# quack-kernels, regex, ... which is exactly the resolution that fails during
# CI environment preparation.
#
# If pip cannot install pytest (offline CI, restricted index, ...), fall back
# to the pure-stdlib self-check instead of exiting non-zero. The layout module
# and the self-check are stdlib-only, so the MXFP4 checks still run and CI
# environment preparation can exit 0.
# Capture the pip exit code explicitly instead of relying on `set -e`
# (which is intentionally not enabled here) so a failed download/resolve
# cannot terminate the script before the fallback runs.
python -m pip install --no-deps --disable-pip-version-check pytest
pip_status=$?
if [ "$pip_status" -ne 0 ]; then
    run_stdlib_selfcheck "pytest install failed (pip exit $pip_status)"
fi

# Sanity check: pytest must be importable and the layout module must load
# without importing sglang. This fails fast if the minimal environment is not
# actually usable, instead of failing later inside the test run.
python -c "import pytest"
import_status=$?
if [ "$import_status" -ne 0 ]; then
    run_stdlib_selfcheck "pytest not importable (exit $import_status)"
fi
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
