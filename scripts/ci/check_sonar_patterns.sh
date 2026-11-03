#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Static grep checks for the patterns fixed by the upstream sonar PR.
# Fails if any of the targeted files still contain the anti-patterns.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

status=0

check() {
  local desc="$1"
  shift
  if "$@"; then
    echo "FAIL: ${desc}"
    status=1
  else
    echo "OK: ${desc}"
  fi
}

# Mutable default arguments in the quantization utils module.
check "no mutable dict/list/set defaults in quantization/utils.py" \
  grep -nE 'def [a-zA-Z_]+\([^)]*=\s*(\{\}|\[\]|set\(\))' \
  python/sglang/srt/layers/quantization/utils.py

# hasattr on dataclass optional fields in the inkling quantization config.
check "no hasattr on dataclass fields in inkling quantization config" \
  grep -nE 'hasattr\([^,]+,\s*"(vocab_embedding|lm_head)"\)' \
  python/sglang/srt/models/inkling_common/quantization/config.py

exit "${status}"
