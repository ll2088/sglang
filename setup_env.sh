#!/usr/bin/env bash
# Root environment-preparation entry for the run_suite verification jobs.
#
# This script must NEVER resolve the full sglang dependency set. It only
# delegates to test/unit/setup_env.sh, which installs the pytest-only
# requirements file. Any editable install request (`pip install -e python`,
# `--editable=./python`, ...) is rejected before delegation so CI cannot fall
# through into a full dependency resolution that fails during setup.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT_SETUP="${SCRIPT_DIR}/test/unit/setup_env.sh"

# Explicit entry marker: CI logs must show this line to prove the lightweight
# entry was hit instead of a direct `pip install -e python`.
echo "[setup_env] entry marker: ${BASH_SOURCE[0]}"
echo "[setup] resolved root=${SCRIPT_DIR}"
echo "[setup] resolved unit setup=${UNIT_SETUP}"

# --self-check: prove the entry is reachable and the guard is wired without
# installing anything. Used as a CI pre-flight gate.
if [ "${1:-}" = "--self-check" ]; then
    echo "[setup_env] self-check: entry reachable, no install performed"
    echo "[setup_env] self-check: unit setup exists=$([ -f "${UNIT_SETUP}" ] && echo yes || echo no)"
    exit 0
fi

# Reject editable installs in every form CI may pass. Walk the arguments so
# `-e python`, `-e ./python`, `-e.`, `--editable python`, `--editable=python`
# and a full `pip install -e python` command line are all caught.
_reject_editable() {
    echo "[setup_env] ERROR: editable install requested ($*)." >&2
    echo "[setup_env] Use the pytest-only entry instead: bash test/unit/setup_env.sh" >&2
    echo "[setup_env] Refusing to resolve the full sglang dependency set." >&2
    exit 2
}

_args=("$@")
_i=0
while [ "${_i}" -lt "${#_args[@]}" ]; do
    _arg="${_args[${_i}]}"
    case "${_arg}" in
        -e|--editable)
            _reject_editable "${_args[*]}"
            ;;
        -e=*|--editable=*)
            _reject_editable "${_args[*]}"
            ;;
        -e./*|-e.|-e../*)
            _reject_editable "${_args[*]}"
            ;;
        pip)
            # A full `pip install ...` command line: scan the rest for -e.
            for _rest in "${_args[@]:$((_i + 1))}"; do
                case "${_rest}" in
                    -e|--editable|-e=*|--editable=*|-e./*|-e.|-e../*)
                        _reject_editable "${_args[*]}"
                        ;;
                esac
            done
            ;;
    esac
    _i=$((_i + 1))
done

if [ ! -f "${UNIT_SETUP}" ]; then
    echo "[setup_env] ERROR: missing ${UNIT_SETUP}" >&2
    exit 2
fi

echo "[setup_env] delegating to ${UNIT_SETUP}"
exec bash "${UNIT_SETUP}"
