#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Hermetic test-dependency installer for the quantization regression suite.
#
# Goals:
#   * never fall back to the system interpreter (/usr/lib/python3/dist-packages)
#   * never resolve the full sglang dependency tree (use --no-deps)
#   * pin versions through a constraints file so wheel tags stay consistent
#
# Usage: scripts/ci/install_test_deps.sh [constraints-file]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
CONSTRAINTS_FILE="${1:-${REPO_ROOT}/scripts/ci/constraints-ci.txt}"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "ERROR: ${VENV_DIR} does not exist; create the venv before running this script" >&2
  exit 1
fi

PYTHON_BIN="${VENV_DIR}/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: ${PYTHON_BIN} is missing or not executable" >&2
  exit 1
fi

# 1) Force the interpreter to the repo venv and fail loudly otherwise.
ACTUAL_PREFIX="$("${PYTHON_BIN}" -c 'import sys; print(sys.prefix)')"
if [[ "${ACTUAL_PREFIX}" != "${VENV_DIR}" ]]; then
  echo "ERROR: interpreter prefix is '${ACTUAL_PREFIX}', expected '${VENV_DIR}'" >&2
  exit 1
fi

PY_VERSION="$("${PYTHON_BIN}" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
echo "Using ${PYTHON_BIN} (Python ${PY_VERSION}, prefix ${ACTUAL_PREFIX})"

# 2) Refuse to run if the interpreter can see the distro site-packages.
if "${PYTHON_BIN}" -c 'import sys; sys.exit(0 if any("/usr/lib/python3/dist-packages" in p for p in sys.path) else 1)'; then
  echo "ERROR: distro site-packages leaked into sys.path; recreate the venv" >&2
  exit 1
fi

if [[ ! -f "${CONSTRAINTS_FILE}" ]]; then
  echo "ERROR: constraints file '${CONSTRAINTS_FILE}' not found" >&2
  exit 1
fi

# 3) Install with --no-deps so the sglang dependency tree is never resolved.
export PIP_CONSTRAINT="${CONSTRAINTS_FILE}"
export PIP_DISABLE_PIP_VERSION_CHECK=1

"${PYTHON_BIN}" -m pip install --upgrade pip --no-deps
"${PYTHON_BIN}" -m pip install \
  --no-deps \
  --only-binary=:all: \
  -c "${CONSTRAINTS_FILE}" \
  -r "${REPO_ROOT}/scripts/ci/requirements-test-min.txt"

# 4) Sanity check: the modules the regression tests import must be importable.
"${PYTHON_BIN}" - <<'PY'
import importlib
import sys

required = ["pytest"]
missing = []
for name in required:
    try:
        importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001 - report and fail
        missing.append(f"{name}: {exc}")
if missing:
    print("ERROR: missing test dependencies:", file=sys.stderr)
    for item in missing:
        print(f"  - {item}", file=sys.stderr)
    raise SystemExit(1)
print("test dependencies OK")
PY
