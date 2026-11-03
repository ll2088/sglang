#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Thin wrapper around check_static_patterns.py so CI can call a stable path.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python "${SCRIPT_DIR}/check_static_patterns.py"
