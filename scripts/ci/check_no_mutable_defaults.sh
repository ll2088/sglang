#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Thin wrapper around check_static_patterns.py so that CI can call a
# single shell entry point for the mutable-default / hasattr / pickle
# checks.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "${SCRIPT_DIR}/check_static_patterns.py" "$@"
