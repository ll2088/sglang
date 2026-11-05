#!/usr/bin/env bash
# Thin wrapper around scripts/ci/setup_extensible_kv_cache_env.sh.
# exec keeps the exit code and avoids path ambiguity between the two copies.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${REPO_ROOT}/scripts/ci/setup_extensible_kv_cache_env.sh"

if [ ! -f "${SCRIPT}" ]; then
    printf 'ERROR: %s not found\n' "${SCRIPT}" >&2
    exit 1
fi

exec bash "${SCRIPT}" "$@"
