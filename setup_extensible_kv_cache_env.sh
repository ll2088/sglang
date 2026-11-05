#!/usr/bin/env bash
# Thin wrapper kept for callers that reference the repository-root path.
# The real setup lives in scripts/ci/setup_extensible_kv_cache_env.sh so
# there is a single source of truth.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$REPO_ROOT/scripts/ci/setup_extensible_kv_cache_env.sh" "$@"
