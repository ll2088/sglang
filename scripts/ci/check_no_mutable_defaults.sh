#!/usr/bin/env bash
# Fail if the files touched by this change still contain mutable default
# arguments (dict/list/set literals in a function signature).
set -euo pipefail

FILES=(
  "python/sglang/srt/layers/quantization/utils.py"
  "python/sglang/srt/layers/quantization/compressed_tensors/utils.py"
  "python/sglang/srt/layers/quantization/quark/utils.py"
  "python/sglang/srt/models/inkling_common/quantization/config.py"
  "python/sglang/srt/utils/hf_transformers/config.py"
  "python/sglang/srt/model_loader/utils.py"
)

status=0
for f in "${FILES[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "SKIP (missing): $f"
    continue
  fi
  # Match `= {}`, `= []`, `= set()` inside a def signature line.
  if grep -nE 'def .*=[[:space:]]*(\{\}|\[\]|set\(\))' "$f"; then
    echo "ERROR: mutable default argument found in $f" >&2
    status=1
  fi
done

if [[ $status -eq 0 ]]; then
  echo "OK: no mutable default arguments found"
fi
exit $status
