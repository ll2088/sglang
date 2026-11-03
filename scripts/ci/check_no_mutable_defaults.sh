#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Grep-based guard against mutable default arguments in the quantization,
# model loader and config modules touched by this port.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

TARGETS=(
  "${REPO_ROOT}/python/sglang/srt/layers/quantization"
  "${REPO_ROOT}/python/sglang/srt/models/inkling_common"
  "${REPO_ROOT}/python/sglang/srt/utils/hf_transformers"
  "${REPO_ROOT}/python/sglang/srt/model_loader"
)

status=0
for target in "${TARGETS[@]}"; do
  [[ -d "${target}" ]] || continue
  # Match `def f(..., x={...})`, `x=[...]`, `x=set()` in signatures.
  if grep -RInE 'def [A-Za-z_][A-Za-z0-9_]*\([^)]*=[[:space:]]*(\{|\[|set\(\))' "${target}" --include='*.py'; then
    echo "[check_no_mutable_defaults] mutable default found under ${target}" >&2
    status=1
  fi
done

if [[ ${status} -eq 0 ]]; then
  echo "[check_no_mutable_defaults] passed"
fi
exit ${status}
