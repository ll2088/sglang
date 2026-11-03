#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Lightweight test dependency installer for the quantization regression tests.
#
# These tests only need the Python standard library and pytest, so we avoid
# installing the full sglang dependency tree (which pulls heavy packages such
# as py-spy, quack-kernels, etc. and can fail on newer Python versions).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONSTRAINTS="${REPO_ROOT}/constraints-ci.txt"

if [[ ! -f "${CONSTRAINTS}" ]]; then
  echo "[install_test_deps] missing constraints file: ${CONSTRAINTS}" >&2
  exit 1
fi

echo "[install_test_deps] using constraints file: ${CONSTRAINTS}"
echo "[install_test_deps] python: $(python -V 2>&1)"

# Only install what the lightweight regression tests need. Pin pytest via the
# constraints file so resolution stays deterministic across runners.
python -m pip install --upgrade pip
python -m pip install --only-binary=:all: -c "${CONSTRAINTS}" pytest

echo "[install_test_deps] done"
