#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Static check: reject mutable default arguments in Python function signatures.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PATTERN='def [A-Za-z_][A-Za-z0-9_]*\([^)]*=[[:space:]]*(\{|\[|set\(|dict\(|list\()'

if grep -RInE "${PATTERN}" "${REPO_ROOT}/python" "${REPO_ROOT}/test" 2>/dev/null; then
  echo "ERROR: mutable default argument detected." >&2
  exit 1
fi

echo "No mutable default arguments detected."
