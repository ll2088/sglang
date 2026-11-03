#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Static grep check: reject mutable default arguments in Python sources.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

PATTERN='def [A-Za-z_][A-Za-z0-9_]*\([^)]*=[[:space:]]*(\{|\[|set\(|dict\(|list\()'

if grep -rnE "${PATTERN}" --include='*.py' python/ test/ 2>/dev/null; then
  echo "error: mutable default argument detected" >&2
  exit 1
fi

echo "no mutable default arguments found"
