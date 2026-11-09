#!/usr/bin/env bash
# Root environment-preparation entry for the run_suite verification jobs.
#
# These checks only need pytest: test/unit/conftest.py stubs out
# sglang.test.ci.* so the full sglang runtime (torch, kernels, ...) is not
# required. CI must call this script instead of `pip install -e python`,
# which resolves the full sglang dependency set and can fail before any test
# runs.
#
# Usage (from anywhere):
#
#     bash setup_env.sh
#
# Any editable install request is rejected explicitly so CI cannot silently
# fall back to the full dependency resolution.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT_SETUP="${SCRIPT_DIR}/test/unit/setup_env.sh"

for arg in "$@"; do
    case "${arg}" in
        -e|--editable|--editable=*|-e=*)
            echo "[setup] refusing editable install (${arg}): this entry installs" >&2
            echo "[setup] only test/unit/requirements.txt (pytest). Use" >&2
            echo "[setup] 'pip install -e python' explicitly if you really need" >&2
            echo "[setup] the full sglang dependency set." >&2
            exit 2
            ;;
    esac
done

if [ ! -f "${UNIT_SETUP}" ]; then
    echo "[setup] missing ${UNIT_SETUP}" >&2
    exit 2
fi

echo "[setup] delegating to ${UNIT_SETUP}"
exec bash "${UNIT_SETUP}" "$@"
