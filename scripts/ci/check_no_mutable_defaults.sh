#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Thin wrapper so CI can call the static pattern checker from a shell step.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "${SCRIPT_DIR}/check_static_patterns.py" "$@"
