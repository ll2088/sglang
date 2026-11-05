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
# Sole installation entry point for the extensible KV cache CI dimensions.
#
# The CI runner ships a system python3.14; installing the package with it
# resolves cp314 wheels and breaks the python3.12 assertions in the test
# commands. This script therefore creates an isolated python3.12 venv and
# performs every install through it. It must be the only install step run by
# the tests / stability / performance / usability dimensions.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# Never let an inherited PYTHONPATH leak the system interpreter's packages
# into the venv.
unset PYTHONPATH || true

if ! command -v python3.12 >/dev/null 2>&1; then
    echo "ERROR: python3.12 not found on PATH" >&2
    command -v python3 || true
    python3 -V || true
    exit 1
fi

rm -rf .venv
python3.12 -m venv .venv

VENV_PY="${REPO_ROOT}/.venv/bin/python"
VENV_PIP="${REPO_ROOT}/.venv/bin/python -m pip"

# Fail fast if the venv is not the interpreter the test commands assert on.
"${VENV_PY}" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'

# Only binary wheels: no source builds, and no cp314 wheels can be selected
# because the interpreter is 3.12.
export PIP_ONLY_BINARY=:all:
export PIP_DISABLE_PIP_VERSION_CHECK=1

${VENV_PIP} install --upgrade pip setuptools wheel

# Dry run first so a cp314 resolution fails before anything is installed.
DRY_RUN_LOG="$(mktemp)"
trap 'rm -f "${DRY_RUN_LOG}"' EXIT
if ! ${VENV_PIP} install --dry-run -e python >"${DRY_RUN_LOG}" 2>&1; then
    echo "ERROR: pip dry-run failed" >&2
    cat "${DRY_RUN_LOG}" >&2
    "${VENV_PY}" -V || true
    ${VENV_PIP} -V || true
    ${VENV_PIP} debug --verbose || true
    exit 1
fi
if grep -q 'cp314' "${DRY_RUN_LOG}"; then
    echo "ERROR: pip resolved cp314 wheels; refusing to install" >&2
    grep 'cp314' "${DRY_RUN_LOG}" >&2 || true
    exit 1
fi

${VENV_PIP} install -e python

# Post-install assertions: interpreter version and importability.
if ! "${VENV_PY}" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'; then
    echo "ERROR: venv interpreter is not python3.12" >&2
    "${VENV_PY}" -V || true
    ${VENV_PIP} -V || true
    ${VENV_PIP} debug --verbose || true
    exit 1
fi
if ! "${VENV_PY}" -c 'import sglang'; then
    echo "ERROR: sglang is not importable in the venv" >&2
    "${VENV_PY}" -V || true
    ${VENV_PIP} -V || true
    ${VENV_PIP} debug --verbose || true
    exit 1
fi

echo "extensible KV cache CI environment ready: $("${VENV_PY}" -V)"
