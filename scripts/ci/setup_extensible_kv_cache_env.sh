#!/usr/bin/env bash
# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Prepare an isolated Python 3.12 virtualenv for the extensible KV cache tests.
#
# Every pip invocation goes through "$VENV/bin/python -m pip --isolated" so the
# install can never fall back to the system dist-packages or to a pip that is
# bound to a different interpreter. The script fails loudly with diagnostics
# instead of silently producing a broken environment.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="${REPO_ROOT}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

log() { printf '[setup_extensible_kv_cache_env] %s\n' "$*" >&2; }

die() {
  log "ERROR: $*"
  log "--- diagnostics ---"
  log "which python3.12: $(command -v python3.12 || echo '<missing>')"
  log "which pip: $(command -v pip || echo '<missing>')"
  log "PIP_TARGET=${PIP_TARGET-<unset>}"
  log "PIP_PREFIX=${PIP_PREFIX-<unset>}"
  log "PIP_INDEX_URL=${PIP_INDEX_URL-<unset>}"
  log "PYTHONPATH=${PYTHONPATH-<unset>}"
  log "PYTHONHOME=${PYTHONHOME-<unset>}"
  if [ -x "${VENV}/bin/python" ]; then
    log "venv python: $("${VENV}/bin/python" -c 'import sys; print(sys.executable, sys.version)' 2>&1 || true)"
    log "venv pip: $("${VENV}/bin/python" -m pip --version 2>&1 || true)"
  fi
  exit 1
}

# Drop environment variables that redirect installs away from the venv.
unset PIP_TARGET PIP_PREFIX PIP_ROOT_USER_ACTION PYTHONPATH PYTHONHOME || true
unset PIP_REQUIRE_VIRTUALENV || true

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  die "${PYTHON_BIN} not found on PATH"
fi

if [ ! -x "${VENV}/bin/python" ]; then
  log "creating venv at ${VENV}"
  "${PYTHON_BIN}" -m venv "${VENV}" || die "failed to create venv"
fi

VENV_PY="${VENV}/bin/python"

# 1) The venv interpreter must be Python 3.12 and must be the one we invoke.
"${VENV_PY}" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' \
  || die "venv interpreter is not Python 3.12"

# 2) pip must resolve to the venv, not to system dist-packages.
PIP_VERSION="$("${VENV_PY}" -m pip --version 2>&1)" || die "venv pip is not usable"
log "${PIP_VERSION}"
case "${PIP_VERSION}" in
  *"${VENV}"*) ;;
  *) die "pip does not belong to ${VENV}: ${PIP_VERSION}" ;;
esac

# 3) Make sure the venv site-packages is the only writable target.
"${VENV_PY}" - <<'PY' || die "venv isolation check failed"
import sys
import sysconfig

prefix = sys.prefix
purelib = sysconfig.get_paths()["purelib"]
assert purelib.startswith(prefix), (purelib, prefix)
for entry in sys.path:
    if "dist-packages" in entry and not entry.startswith(prefix):
        raise SystemExit(f"system dist-packages leaked into sys.path: {entry}")
print(f"isolated venv prefix={prefix} purelib={purelib}")
PY

# 4) Install with an explicit interpreter and --isolated so pip ignores
#    user/global config and never falls back to a different ABI.
log "upgrading pip/setuptools/wheel inside the venv"
"${VENV_PY}" -m pip --isolated install --upgrade pip setuptools wheel \
  || die "failed to upgrade pip/setuptools/wheel"

log "installing sglang (editable) with --only-binary=:all:"
"${VENV_PY}" -m pip --isolated install --only-binary=:all: -e "${REPO_ROOT}/python" \
  || die "failed to install sglang into the venv"

# 5) Smoke test imports and print a short freeze summary.
"${VENV_PY}" - <<'PY' || die "import smoke test failed"
import sys
assert sys.version_info[:2] == (3, 12), sys.version
import torch  # noqa: F401
import sglang  # noqa: F401
print(f"smoke ok: python={sys.version.split()[0]} torch={torch.__version__}")
PY

log "pip freeze (head):"
"${VENV_PY}" -m pip --isolated freeze | head -n 40 >&2 || true

log "environment ready at ${VENV}"
