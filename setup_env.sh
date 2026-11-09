#!/usr/bin/env bash
# Repository-root environment preparation entry.
#
# CI environment preparation for the run_suite verification dimensions must not
# resolve the full sglang dependency set (`pip install -e python`), which pulls
# torch/kernels and can fail before any test runs. This entry delegates to the
# pytest-only setup script so whichever path CI calls, the lightweight install
# is the one that actually runs.
#
# Usage (from the repository root):
#
#     bash setup_env.sh
set -euo pipefail

echo "[setup] using $(realpath "$0")"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${SCRIPT_DIR}/test/unit/setup_env.sh"

if [ ! -f "${TARGET}" ]; then
    echo "[setup] missing ${TARGET}" >&2
    exit 1
fi

echo "[setup] delegating to ${TARGET}"
exec bash "${TARGET}" "$@"
