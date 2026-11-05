#!/usr/bin/env bash
# Thin wrapper; the only implementation lives in scripts/ci/.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${REPO_ROOT}/scripts/ci/setup_extensible_kv_cache_env.sh" "$@"
