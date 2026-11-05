#!/usr/bin/env bash
# Unique environment setup entry for the extensible KV cache checks.
# Creates an isolated python3.12 virtualenv and installs the project into it.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3.12}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3.12 || true)"
fi
if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: python3.12 not found; set PYTHON_BIN to an absolute path" >&2
  exit 1
fi

VENV_DIR="${REPO_ROOT}/.venv"
rm -rf "${VENV_DIR}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"

VENV_PY="${VENV_DIR}/bin/python"
VENV_PIP="${VENV_DIR}/bin/pip"

# Hard assertions: interpreter version and pip must live inside the venv.
"${VENV_PY}" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
PIP_PATH="$("${VENV_PY}" -m pip -V)"
case "${PIP_PATH}" in
  "${VENV_DIR}"*) ;;
  *)
    echo "ERROR: pip does not resolve inside ${VENV_DIR}: ${PIP_PATH}" >&2
    exit 1
    ;;
esac

# Never fall back to system dist-packages.
export PIP_REQUIRE_VIRTUALENV=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

"${VENV_PY}" -m pip install --no-cache-dir --upgrade pip setuptools wheel
"${VENV_PY}" -m pip install --no-cache-dir -e "${REPO_ROOT}"

# Import smoke test plus diagnostics.
"${VENV_PY}" - <<'PY'
import sys
import sglang
import sglang.srt.mem_cache.vmm_allocator as vmm
import sglang.srt.mem_cache.extensible_kv_cache as ekv
print("python:", sys.version)
print("executable:", sys.executable)
print("sys.path:", sys.path)
print("vmm_allocator:", vmm.__file__)
print("extensible_kv_cache:", ekv.__file__)
PY

"${VENV_PY}" -m pip -V
