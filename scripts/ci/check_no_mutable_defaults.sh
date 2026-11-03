#!/usr/bin/env bash
# Static check: reject mutable default arguments (dict/list/set literals) in
# function signatures under the quantization and model-loading modules.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

TARGETS=(
  "${REPO_ROOT}/python/sglang/srt/layers/quantization"
  "${REPO_ROOT}/python/sglang/srt/model_loader"
  "${REPO_ROOT}/python/sglang/srt/utils/hf_transformers"
)

status=0
for target in "${TARGETS[@]}"; do
  if [[ ! -d "${target}" ]]; then
    continue
  fi
  # Match `= {}`, `= []`, `= set()` inside a def signature line.
  if grep -RInE 'def [^(]+\([^)]*= *(\{\}|\[\]|set\(\))' "${target}" --include='*.py'; then
    echo "error: mutable default argument found under ${target}" >&2
    status=1
  fi
done

if [[ "${status}" -eq 0 ]]; then
  echo "no mutable default arguments found"
fi
exit "${status}"
