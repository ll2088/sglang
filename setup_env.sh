#!/usr/bin/env bash
# Lightweight environment-preparation entry for the run_suite verification
# dimensions. CI must call this script (or test/unit/setup_env.sh) instead of
# `pip install -e python`, which resolves the full sglang dependency set and
# can fail before any test runs.
#
# This script never installs anything itself: it rejects editable installs and
# delegates to test/unit/setup_env.sh, which installs only
# test/unit/requirements.txt (pytest).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT_SETUP="${SCRIPT_DIR}/test/unit/setup_env.sh"

echo "[setup_env] lightweight entry: ${SCRIPT_DIR}/setup_env.sh"
echo "[setup_env] delegating to ${UNIT_SETUP}"

# Reject every editable-install form CI may pass, including the short form,
# the `-e./python` glued form and the `--editable=...` form. These would
# resolve the full sglang dependency set instead of the pytest-only one.
for arg in "$@"; do
    case "$arg" in
        -e|--editable|-e./*|-e=*|--editable=*)
            echo "[setup_env] error: editable installs are not supported here" >&2
            echo "[setup_env] use 'bash test/unit/setup_env.sh' (pytest only)" >&2
            exit 2
            ;;
    esac
done

if [ ! -f "${UNIT_SETUP}" ]; then
    echo "[setup_env] error: missing ${UNIT_SETUP}" >&2
    exit 2
fi

exec bash "${UNIT_SETUP}" "$@"
