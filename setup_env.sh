#!/usr/bin/env bash
# Lightweight environment preparation entry for the run_suite verification
# dimensions. It must never resolve the full sglang dependency set: the unit
# tests under test/unit only need pytest, and `pip install -e python` pulls in
# torch/kernels/... which can fail before any test runs.
#
# Any editable-install request (the historical CI command) is rejected with a
# clear diagnostic and a non-zero exit code, then the pytest-only unit script
# is delegated to.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT_SETUP="${SCRIPT_DIR}/test/unit/setup_env.sh"

[setup] using "${SCRIPT_DIR}/setup_env.sh"
[setup] delegating to "${UNIT_SETUP}"

# Reject editable installs before anything can reach pip. Match the long and
# short forms, `python -m pip install -e ...`, and `-e ./python` variants.
for arg in "$@"; do
    case "${arg}" in
        -e|--editable|-e=*|--editable=*|-e./*|--editable./*)
            echo "[setup] error: editable installs are not supported here." >&2
            echo "[setup] use 'bash test/unit/setup_env.sh' (pytest only) instead of 'pip install -e python'." >&2
            exit 2
            ;;
    esac
done

if [ ! -f "${UNIT_SETUP}" ]; then
    echo "[setup] error: missing ${UNIT_SETUP}" >&2
    exit 2
fi

exec bash "${UNIT_SETUP}" "$@"
