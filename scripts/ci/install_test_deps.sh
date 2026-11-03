#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Install the minimal dependency set needed by the quant regression tests.
#
# This script deliberately avoids resolving sglang's full dependency tree:
# it installs with --no-deps and pins versions via constraints-ci.txt so the
# CI environment stays small and reproducible.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ ! -f .venv/bin/activate ]]; then
  echo "error: .venv not found; create it before running this script" >&2
  exit 1
fi

# shellcheck disable=SC1091
. .venv/bin/activate

python - <<'PY'
import sys
if sys.prefix == sys.base_prefix:
    raise SystemExit("error: .venv is not active")
print(f"using interpreter: {sys.executable}")
PY

CONSTRAINTS="${REPO_ROOT}/scripts/ci/constraints-ci.txt"
REQUIREMENTS="${REPO_ROOT}/scripts/ci/requirements-test-min.txt"

if [[ ! -f "${CONSTRAINTS}" ]]; then
  echo "error: missing ${CONSTRAINTS}" >&2
  exit 1
fi
if [[ ! -f "${REQUIREMENTS}" ]]; then
  echo "error: missing ${REQUIREMENTS}" >&2
  exit 1
fi

# Fail loudly with the full pip output instead of a truncated log.
set -x
python -m pip install --upgrade pip
python -m pip install --no-deps -c "${CONSTRAINTS}" -r "${REQUIREMENTS}"
set +x

echo "minimal test dependencies installed"
