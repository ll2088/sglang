#!/usr/bin/env bash
# Grep-based guard for the patterns fixed in this change set.
# Exits non-zero if a forbidden pattern is found in the touched files.
set -euo pipefail

FILES=(
  python/sglang/srt/layers/quantization/utils.py
  python/sglang/srt/layers/quantization/compressed_tensors/utils.py
  python/sglang/srt/layers/quantization/quark/utils.py
  python/sglang/srt/models/inkling_common/quantization/config.py
  python/sglang/srt/utils/hf_transformers/config.py
  python/sglang/srt/model_loader/utils.py
)

status=0

# Mutable default arguments: `= {}`, `= []`, `= set()` in signatures.
if grep -nE '=\s*(\{\}|\[\]|set\(\))' "${FILES[@]}"; then
  echo "ERROR: mutable default argument found" >&2
  status=1
fi

# hasattr on dataclass optional fields is not reliable; flag the pattern.
if grep -nE 'hasattr\([^,]+,\s*"(vocab_embedding|lm_head)"' "${FILES[@]}"; then
  echo "ERROR: hasattr used on dataclass optional field" >&2
  status=1
fi

exit "${status}"
