#!/usr/bin/env bash
# Lightweight environment preparation entry for the run_suite verification
# dimensions (collect-diagnostics / exit-codes / tests / usability).
#
# CI previously ran `pip install -e python`, which resolves the full sglang
# dependency set (torch, kernels, py-spy, quack-kernels, ...) and can fail
# before any test runs. This script installs only test/unit/requirements.txt
# (pytest) and self-checks pytest so an environment problem is reported as an
# environment failure instead of being mistaken for a test failure.
#
# It is intentionally cwd-independent and rejects editable sglang installs so
# the full dependency set cannot silently come back.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")" && pwd)"
REQUIREMENTS="${SCRIPT_DIR}/test/unit/requirements.txt"

# Reject any attempt to route this entry through an editable sglang install.
for arg in "$@"; do
    case "${arg}" in
        -e|--editable|-e=*|--editable=*)
            echo "[setup] ERROR: editable installs are not supported by this entry" >&2
            echo "[setup] use test/unit/requirements.txt (pytest only) instead" >&2
            exit 2
            ;;
    esac
done

if [[ ! -f "${REQUIREMENTS}" ]]; then
    echo "[setup] ERROR: missing ${REQUIREMENTS}" >&2
    exit 2
fi

# Guard against the full sglang dependency set sneaking back in.
if grep -Eq '^(sglang==|quack-kernels)' "${REQUIREMENTS}"; then
    echo "[setup] ERROR: ${REQUIREMENTS} must stay pytest-only" >&2
    exit 2
fi

echo "[setup] using ${REQUIREMENTS}"
python -m pip install --disable-pip-version-check -r "${REQUIREMENTS}"

python - <<'PY'
import pytest
print("[setup] pytest", pytest.__version__)
PY
python -m pytest --version
