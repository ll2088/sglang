#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Static grep checks for the patterns fixed by the upstream PR:
#   - mutable default arguments (dict/list/set literals in signatures)
#   - dataclass optional fields checked with hasattr
#
# This is a best-effort check; it is intentionally conservative and only
# reports obvious matches so that it can run in CI without false positives.
set -euo pipefail

ROOT="${1:-python/sglang/srt/layers/quantization}"

status=0

if [[ ! -d "${ROOT}" ]]; then
  echo "error: ${ROOT} is not a directory" >&2
  exit 1
fi

echo "checking mutable defaults under ${ROOT}"
if grep -RInE 'def [A-Za-z_][A-Za-z0-9_]*\([^)]*=[[:space:]]*(\{|\[|set\()' "${ROOT}" 2>/dev/null; then
  echo "error: mutable default argument detected" >&2
  status=1
fi

echo "checking hasattr on dataclass fields under ${ROOT}"
if grep -RInE 'hasattr\([^,]+,[[:space:]]*[\"'\''][A-Za-z_][A-Za-z0-9_]*[\"'\'']\)' "${ROOT}" 2>/dev/null | grep -v 'getattr' >/dev/null; then
  echo "warning: hasattr usage found; review dataclass optional fields manually" >&2
fi

if [[ "${status}" -ne 0 ]]; then
  exit "${status}"
fi

echo "static checks passed"
