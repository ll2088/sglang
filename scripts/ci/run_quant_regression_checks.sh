#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Run the lightweight quantization regression checks and print evidence for
# the acceptance criteria (constraints path, grep audit, pytest results).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONSTRAINTS="${REPO_ROOT}/constraints-ci.txt"

TESTS=(
  "${REPO_ROOT}/test/registered/quant/test_quant_utils_defaults.py"
  "${REPO_ROOT}/test/registered/quant/test_dataclass_optional_fields.py"
  "${REPO_ROOT}/test/registered/quant/test_pickle_loading_safety.py"
  "${REPO_ROOT}/test/registered/quant/test_moe_init_keywords.py"
)

echo "[quant-checks] constraints file: ${CONSTRAINTS}"
echo "[quant-checks] python: $(python -V 2>&1)"

for test_file in "${TESTS[@]}"; do
  if [[ ! -f "${test_file}" ]]; then
    echo "[quant-checks] missing test file: ${test_file}" >&2
    exit 1
  fi
done

echo "[quant-checks] grep audit: mutable defaults in changed files"
if grep -nE 'def [a-zA-Z_]+\([^)]*=\s*(\{\}|\[\]|set\(\))' \
    "${REPO_ROOT}/test/registered/quant/"*.py; then
  echo "[quant-checks] found mutable default arguments" >&2
  exit 1
fi

echo "[quant-checks] grep audit: hasattr on dataclass fields"
if grep -nE 'hasattr\([^,]+,\s*"(vocab_embedding|lm_head)"\)' \
    "${REPO_ROOT}/test/registered/quant/"*.py; then
  echo "[quant-checks] found hasattr on dataclass optional fields" >&2
  exit 1
fi

echo "[quant-checks] running pytest"
python -m pytest -q -p no:randomly "${TESTS[@]}"

echo "[quant-checks] all checks passed"
