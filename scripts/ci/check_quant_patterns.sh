#!/usr/bin/env bash
# Static review for the defect patterns targeted by the upstream port:
#   1. mutable default arguments (dict/list/set literals in signatures)
#   2. in-place mutation of module-level mapping constants
#   3. hasattr() used on dataclass optional fields
# Exits non-zero when a pattern is found so CI can surface it.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TARGET_DIR="${REPO_ROOT}/python/sglang/srt/layers/quantization"

status=0

if [ ! -d "${TARGET_DIR}" ]; then
  echo "Target directory not found: ${TARGET_DIR}" >&2
  exit 1
fi

echo "[1/3] checking mutable default arguments"
if grep -rnE 'def [A-Za-z_][A-Za-z0-9_]*\([^)]*=[[:space:]]*(\{|\[|set\()' "${TARGET_DIR}" --include='*.py'; then
  echo "ERROR: mutable default argument found" >&2
  status=1
fi

echo "[2/3] checking in-place mutation of module-level mappings"
if grep -rnE '^[A-Z_][A-Z0-9_]*\.(update|setdefault|pop|clear)\(' "${TARGET_DIR}" --include='*.py'; then
  echo "ERROR: in-place mutation of module-level mapping found" >&2
  status=1
fi

echo "[3/3] checking hasattr on dataclass optional fields"
if grep -rnE 'hasattr\([^,]+,[[:space:]]*"(vocab_embedding|lm_head)"\)' "${TARGET_DIR}" --include='*.py'; then
  echo "ERROR: hasattr used on dataclass optional field" >&2
  status=1
fi

if [ "${status}" -eq 0 ]; then
  echo "All static checks passed."
fi
exit "${status}"
