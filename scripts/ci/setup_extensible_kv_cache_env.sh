#!/usr/bin/env bash
# CI entry point for the extensible KV cache test environment.
# Delegates to the repository-root setup script so local and CI runs share
# exactly one implementation of the venv/pip constraints.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [ ! -x "$ROOT_DIR/setup_extensible_kv_cache_env.sh" ]; then
  echo "[setup-extensible-kv-cache] ERROR: missing $ROOT_DIR/setup_extensible_kv_cache_env.sh" >&2
  exit 1
fi

exec "$ROOT_DIR/setup_extensible_kv_cache_env.sh" "$@"
