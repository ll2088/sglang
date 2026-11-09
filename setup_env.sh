#!/usr/bin/env bash
# Lightweight environment-preparation entry for the run_suite verification
# jobs. It must never resolve the full sglang dependency set: the unit tests
# under test/unit only need pytest, and `pip install -e python` pulls in torch,
# kernels and the rest, which fails before pytest ever starts.
#
# Any editable-install request is rejected explicitly and the script delegates
# to test/unit/setup_env.sh, which installs only test/unit/requirements.txt.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT_SETUP="${SCRIPT_DIR}/test/unit/setup_env.sh"

# Reject editable installs before doing anything else. Both the short and long
# forms are matched so `setup_env.sh -e` cannot slip past into the full
# dependency resolution.
for arg in "$@"; do
    case "${arg}" in
        -e|--editable|--editable=*|-e=*)
            echo "[setup] error: editable installs are not supported here." >&2
            echo "[setup] 'pip install -e python' resolves the full sglang" >&2
            echo "[setup] dependency set and fails before pytest starts." >&2
            echo "[setup] use the pytest-only entry instead:" >&2
            echo "[setup]   bash test/unit/setup_env.sh" >&2
            exit 2
            ;;
    esac
done

if [ ! -f "${UNIT_SETUP}" ]; then
    echo "[setup] error: missing ${UNIT_SETUP}" >&2
    exit 2
fi

echo "[setup] using ${UNIT_SETUP}"
exec bash "${UNIT_SETUP}" "$@"
