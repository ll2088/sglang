#!/usr/bin/env bash
# Install only the dependencies needed by the lightweight regression tests.
# Avoids resolving the full sglang runtime dependency set (py-spy, quack-kernels, ...).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

python -m pip install --upgrade pip
python -m pip install -c constraints-ci.txt -e . --no-deps
python -m pip install -c constraints-ci.txt pytest numpy torch transformers huggingface_hub compressed-tensors

python -c "import sglang"
python -m pip freeze
