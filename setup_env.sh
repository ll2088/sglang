#!/usr/bin/env bash
# Root environment-preparation entry for the run_suite verification jobs.
#
# This script must stay lightweight: it only installs the pytest-only
# requirements used by test/unit, never the full sglang dependency set.
# `pip install -e python` resolves torch/kernels/... and can fail before any
# test runs, which previously turned every verification dimension into an
# environment failure instead of a test result.
#
# It delegates to test/unit/setup_env.sh, resolving paths from this script's
# own directory so CI can invoke it from any working directory.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
UNIT_SETUP="${SCRIPT_DIR}/test/unit/setup_env.sh"

# Debug output so CI logs prove which entry was hit and where it delegates.
echo "[setup] pwd=$(pwd)"
echo "[setup] script=${BASH_SOURCE[0]}"
echo "[setup] resolved script dir=${SCRIPT_DIR}"
echo "[setup] delegating to ${UNIT_SETUP}"

# Reject any editable install request before doing anything else: the full
# sglang dependency resolution is exactly what this entry must avoid.
for arg in "$@"; do
    case "$arg" in
        -e|--editable|-e=*|--editable=*)
            echo "[setup] error: editable installs are not supported by this entry;" >&2
            echo "[setup] use 'bash test/unit/setup_env.sh' for the pytest-only setup." >&2
            exit 2
            ;;
    esac
done

if [ ! -f "$UNIT_SETUP" ]; then
    echo "[setup] error: missing delegate ${UNIT_SETUP}" >&2
    exit 2
fi

exec bash "$UNIT_SETUP" "$@"
