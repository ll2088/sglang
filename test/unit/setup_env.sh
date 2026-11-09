#!/usr/bin/env bash
# Minimal environment preparation for the run_suite unit tests.
#
# These tests only exercise run_suite.py's exit-code and collection-diagnostic
# branches; conftest.py stubs out sglang.test.ci.* so the full sglang runtime
# (torch, kernels, ...) is not needed. CI verification jobs should call this
# script instead of `pip install -e python`, which resolves the full dependency
# set and can fail during environment preparation.
#
# The script is cwd-independent, prints the resolved path so CI logs prove the
# lightweight entry was hit, and rejects editable sglang installs so the full
# dependency set cannot silently come back.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")" && pwd)"
REQUIREMENTS="${SCRIPT_DIR}/requirements.txt"

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
