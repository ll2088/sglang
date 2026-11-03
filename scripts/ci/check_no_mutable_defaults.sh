#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Grep-based guard against the defect patterns fixed in this change set.
# Exits non-zero when a forbidden pattern is found in the target files.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

TARGETS=(
  "${REPO_ROOT}/python/sglang/srt/layers/quantization/utils.py"
  "${REPO_ROOT}/python/sglang/srt/layers/quantization/compressed_tensors/utils.py"
  "${REPO_ROOT}/python/sglang/srt/layers/quantization/quark/utils.py"
)

status=0

# Mutable default arguments: `= {}`, `= []`, `= set()` in a signature.
for f in "${TARGETS[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "missing target: $f" >&2
    status=1
    continue
  fi
  if grep -nE 'def [^(]*\([^)]*=[[:space:]]*(\{\}|\[\]|set\(\))' "$f"; then
    echo "mutable default argument found in $f" >&2
    status=1
  fi
done

if [[ "${status}" -ne 0 ]]; then
  exit 1
fi

echo "no mutable defaults found"
