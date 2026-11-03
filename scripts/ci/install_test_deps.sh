#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Install the minimal dependency set required by the quantization regression
# tests without resolving the full sglang dependency tree.
set -euxo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONSTRAINTS_ROOT="${REPO_ROOT}/constraints-ci.txt"
CONSTRAINTS_CI="${REPO_ROOT}/scripts/ci/constraints-ci.txt"

python -V
python -m pip -V

if ! diff -u "${CONSTRAINTS_ROOT}" "${CONSTRAINTS_CI}"; then
  echo "constraints-ci.txt files differ; refusing to install" >&2
  exit 1
fi

python -m pip install --no-deps -c "${CONSTRAINTS_CI}" \
  'pytest>=8.0,<9' 'numpy>=1.26,<3' 'torch>=2.1,<3'

python - <<'PY'
import importlib

for name in (
    "sglang.srt.layers.quantization.fp8",
    "sglang.srt.layers.quantization.utils",
    "sglang.srt.layers.quantization.compressed_tensors.utils",
    "sglang.srt.layers.quantization.quark.utils",
    "sglang.srt.models.inkling_common.quantization.config",
    "sglang.srt.utils.hf_transformers.config",
    "sglang.srt.model_loader.utils",
):
    importlib.import_module(name)
print("import smoke check passed")
PY
