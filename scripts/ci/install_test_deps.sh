#!/usr/bin/env bash
# Install the minimal dependency set used by the quant regression tests.
# This intentionally avoids `pip install -e python`, which resolves the full
# sglang runtime dependency tree (py-spy, pyzmq, quack-kernels, ...).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

REQ_FILE="${SCRIPT_DIR}/requirements-test-min.txt"
CONSTRAINTS_FILE="${SCRIPT_DIR}/constraints-ci.txt"

if [[ ! -f "${REQ_FILE}" ]]; then
  echo "ERROR: missing ${REQ_FILE}" >&2
  exit 1
fi
if [[ ! -f "${CONSTRAINTS_FILE}" ]]; then
  echo "ERROR: missing ${CONSTRAINTS_FILE}" >&2
  exit 1
fi

echo "== CI environment =="
python --version || true
python -m pip --version || true
echo "REQ_FILE=${REQ_FILE}"
echo "CONSTRAINTS_FILE=${CONSTRAINTS_FILE}"

PIP_LOG="$(mktemp -t pip-install-XXXXXX.log)"
if ! python -m pip install \
    --disable-pip-version-check \
    --no-input \
    -r "${REQ_FILE}" \
    -c "${CONSTRAINTS_FILE}" \
    >"${PIP_LOG}" 2>&1; then
  echo "ERROR: pip install failed; full log follows" >&2
  cat "${PIP_LOG}" >&2
  echo "== python/pip versions ==" >&2
  python --version >&2 || true
  python -m pip --version >&2 || true
  exit 1
fi
cat "${PIP_LOG}"

# Self-check: pytest must be importable and the target tests must collect.
python -c 'import pytest; print("pytest", pytest.__version__)'

TARGET_TESTS=(
  "${REPO_ROOT}/test/registered/quant/test_quant_utils_defaults.py"
  "${REPO_ROOT}/test/registered/quant/test_dataclass_optional_fields.py"
  "${REPO_ROOT}/test/registered/quant/test_pickle_loading_safety.py"
  "${REPO_ROOT}/test/registered/quant/test_moe_init_keywords.py"
)

for test_file in "${TARGET_TESTS[@]}"; do
  if [[ ! -f "${test_file}" ]]; then
    echo "ERROR: missing test file ${test_file}" >&2
    exit 1
  fi
done

python -m pytest --collect-only -q "${TARGET_TESTS[@]}"
