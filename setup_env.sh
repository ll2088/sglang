#!/usr/bin/env bash
# Repository-root environment preparation entry for the run_suite checks.
#
# CI previously ran `pip install -e python`, which resolves the full sglang
# dependency set (torch, kernels, ...) and can fail before any test runs.
# This script delegates to the pytest-only entry so whichever setup path CI
# calls, it lands on the lightweight install.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIGHTWEIGHT="${SCRIPT_DIR}/test/unit/setup_env.sh"

if [[ ! -f "${LIGHTWEIGHT}" ]]; then
  echo "[setup] ERROR: lightweight setup entry not found: ${LIGHTWEIGHT}" >&2
  exit 1
fi

echo "[setup] using $(realpath "${LIGHTWEIGHT}")"
exec bash "${LIGHTWEIGHT}" "$@"
