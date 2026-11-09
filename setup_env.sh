#!/usr/bin/env bash
# Repository-root environment preparation entry.
#
# Delegates to test/unit/setup_env.sh so whichever setup entry CI calls, only
# test/unit/requirements.txt (pytest) is installed. This avoids the full
# `pip install -e python` dependency resolution, which can fail before any
# test runs.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${REPO_ROOT}/test/unit/setup_env.sh"

echo "[setup] using ${TARGET}"

if [ ! -f "$TARGET" ]; then
  echo "[setup] missing delegate script: $TARGET" >&2
  exit 2
fi

exec bash "$TARGET" "$@"
