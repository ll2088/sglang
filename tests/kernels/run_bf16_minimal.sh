#!/usr/bin/env bash
# Create a minimal venv for the BF16 top-k capability tests and run one of the
# four CI dimensions.  Never falls back to `pip install sglang`: the tests load
# python/sglang/kernels/dsa_topk_bf16_capability.py by explicit path and must
# not import sglang/__init__.py.
#
# Usage: run_bf16_minimal.sh {tests|stability|performance|usability}
set -uo pipefail

DIM="${1:-tests}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv-bf16-minimal}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

log_diagnostics() {
  echo "===== environment diagnostics =====" >&2
  ( "${VENV_DIR}/bin/python" -V || true ) >&2 2>&1
  ( "${VENV_DIR}/bin/python" -m pip --version || true ) >&2 2>&1
  ( "${VENV_DIR}/bin/python" -m pip debug --verbose || true ) 2>&1 | tail -n 50 >&2
  ( "${VENV_DIR}/bin/python" -m pip list || true ) >&2 2>&1
  echo "===== end diagnostics =====" >&2
}

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "error: ${PYTHON_BIN} not found; set PYTHON_BIN to a 3.12 interpreter" >&2
  exit 1
fi

if [ ! -x "${VENV_DIR}/bin/python" ]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}" || { echo "error: venv creation failed" >&2; exit 1; }
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

if ! python -m pip install -v --disable-pip-version-check \
    -r "${REPO_ROOT}/tests/kernels/requirements-minimal.txt" \
    >"${VENV_DIR}/pip-install.log" 2>&1; then
  echo "error: minimal dependency install failed; last 50 lines:" >&2
  tail -n 50 "${VENV_DIR}/pip-install.log" >&2
  log_diagnostics
  exit 1
fi

cd "${REPO_ROOT}" || exit 1

case "${DIM}" in
  tests)
    python -m pytest -q tests/kernels/test_dsa_topk_bf16.py
    ;;
  stability)
    python -m pytest -q -p no:randomly tests/kernels/test_dsa_topk_bf16.py --count=3
    ;;
  performance)
    python -m pytest -q benchmark --durations=10 -k topk
    ;;
  usability)
    python -m pytest -q test -k topk
    ;;
  *)
    echo "error: unknown dimension '${DIM}' (tests|stability|performance|usability)" >&2
    exit 2
    ;;
esac
