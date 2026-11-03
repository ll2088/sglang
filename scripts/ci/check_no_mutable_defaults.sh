#!/usr/bin/env bash
# Grep-based guard against mutable default arguments in the touched files.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

FILES=(
  "python/sglang/srt/layers/quantization/utils.py"
  "python/sglang/srt/layers/quantization/compressed_tensors/utils.py"
  "python/sglang/srt/layers/quantization/quark/utils.py"
  "python/sglang/srt/models/inkling_common/quantization/config.py"
  "python/sglang/srt/utils/hf_transformers/config.py"
  "python/sglang/srt/model_loader/utils.py"
)

status=0
for rel in "${FILES[@]}"; do
  path="${REPO_ROOT}/${rel}"
  [ -f "${path}" ] || continue
  # Match `= {}`, `= []`, `= set()` in function signatures.
  if grep -nE 'def [^(]+\([^)]*=[[:space:]]*(\{\}|\[\]|set\(\))' "${path}"; then
    echo "mutable default found in ${rel}" >&2
    status=1
  fi
done

exit "${status}"
