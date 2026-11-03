#!/usr/bin/env bash
# Install the minimal dependency set required by the quant regression tests.
#
# Design notes:
# - Never run `pip install -e python` without --no-deps: that resolves the
#   full sglang dependency tree (py-spy, pyzmq, quack-kernels, ...) and is the
#   root cause of the previous environment-preparation failures.
# - Print interpreter/pip info up front so version mismatches (e.g. cp314
#   wheels mixed with system dist-packages) are immediately visible.
# - On pip failure, dump the full log instead of a truncated tail.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REQ_FILE="${SCRIPT_DIR}/requirements-test-min.txt"
CONSTRAINTS_FILE="${SCRIPT_DIR}/constraints-ci.txt"
LOG_FILE="${SCRIPT_DIR}/pip-install.log"

PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: interpreter '${PYTHON_BIN}' not found" >&2
  exit 1
fi

echo "=== interpreter info ==="
"${PYTHON_BIN}" -VV || true
command -v "${PYTHON_BIN}" || true
"${PYTHON_BIN}" -m pip --version || true

echo "=== installing minimal test deps ==="
: > "${LOG_FILE}"
if ! "${PYTHON_BIN}" -m pip install \
    --disable-pip-version-check \
    -r "${REQ_FILE}" \
    -c "${CONSTRAINTS_FILE}" \
    >>"${LOG_FILE}" 2>&1; then
  echo "ERROR: pip install of minimal test deps failed; full log follows" >&2
  cat "${LOG_FILE}" >&2
  exit 1
fi

echo "=== installing sglang without deps ==="
if ! "${PYTHON_BIN}" -m pip install \
    --disable-pip-version-check \
    --no-deps \
    -e "${REPO_ROOT}/python" \
    >>"${LOG_FILE}" 2>&1; then
  echo "ERROR: pip install -e python --no-deps failed; full log follows" >&2
  cat "${LOG_FILE}" >&2
  exit 1
fi

echo "=== installed packages ==="
"${PYTHON_BIN}" -m pip list || true

echo "=== self-check: pytest ==="
if ! "${PYTHON_BIN}" -m pytest --version; then
  echo "ERROR: pytest is not usable after install" >&2
  exit 1
fi

echo "=== self-check: collect target tests ==="
TARGET_TESTS=(
  "test/registered/quant/test_quant_utils_defaults.py"
  "test/registered/quant/test_dataclass_optional_fields.py"
  "test/registered/quant/test_pickle_loading_safety.py"
  "test/registered/quant/test_moe_init_keywords.py"
)
EXISTING_TESTS=()
for t in "${TARGET_TESTS[@]}"; do
  if [ -f "${REPO_ROOT}/${t}" ]; then
    EXISTING_TESTS+=("${REPO_ROOT}/${t}")
  fi
done

if [ "${#EXISTING_TESTS[@]}" -eq 0 ]; then
  echo "ERROR: none of the target test files exist" >&2
  exit 1
fi

if ! "${PYTHON_BIN}" -m pytest --collect-only -q "${EXISTING_TESTS[@]}"; then
  echo "ERROR: target tests could not be collected" >&2
  exit 1
fi

echo "=== environment ready ==="
