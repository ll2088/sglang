#!/usr/bin/env bash
# Install the minimal dependency set needed by the self-contained regression
# tests under test/registered/quant/.
#
# The tests deliberately avoid importing sglang, so we install the local
# package with --no-deps to prevent pip from resolving the full runtime
# dependency tree (py-spy, pybase64, pyzmq, quack-kernels, ...).
set -euxo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

python -m pip --version
python -m pip install --upgrade pip

python -m pip install \
    --no-deps \
    -e python \
    -c scripts/ci/constraints-ci.txt \
    -r scripts/ci/requirements-test-min.txt

python -m pytest --version
