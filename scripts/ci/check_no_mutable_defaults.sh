#!/usr/bin/env bash
# Thin wrapper around the AST-based static pattern checker.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec python "${ROOT_DIR}/scripts/ci/check_static_patterns.py"
