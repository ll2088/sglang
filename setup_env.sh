#!/usr/bin/env bash
# Lightweight environment preparation entry for the run_suite verification
# jobs. It must never resolve the full sglang dependency set (torch, kernels,
# py-spy, quack-kernels, ...): those downloads previously failed before any
# test could run, which made every dimension look like a test failure.
#
# This script delegates to test/unit/setup_env.sh, which installs only
# test/unit/requirements.txt (pytest) and self-checks pytest. It also rejects
# editable installs and asserts the install log does not contain the full
# sglang dependency resolution pattern.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIGHTWEIGHT="${SCRIPT_DIR}/test/unit/setup_env.sh"

# Reject any attempt to route this entry through an editable sglang install.
for arg in "$@"; do
    case "${arg}" in
        -e|--editable|-e*|--editable=*)
            echo "[setup] ERROR: editable installs are not supported; " \
                 "use the pytest-only entry instead" >&2
            exit 2
            ;;
    esac
done

if [[ ! -f "${LIGHTWEIGHT}" ]]; then
    echo "[setup] ERROR: missing ${LIGHTWEIGHT}" >&2
    exit 2
fi

echo "[setup] using ${LIGHTWEIGHT}"
exec bash "${LIGHTWEIGHT}" "$@"
