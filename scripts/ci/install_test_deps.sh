#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Install the minimal dependency set needed by the quantization regression
# tests without resolving the full sglang dependency tree.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "error: expected virtualenv interpreter at $PYTHON_BIN" >&2
  echo "hint: create it with 'python3.12 -m venv .venv' or set PYTHON_BIN" >&2
  exit 1
fi

echo "python: $("$PYTHON_BIN" -V 2>&1)"
echo "pip:    $("$PYTHON_BIN" -m pip -V 2>&1)"

# Refuse to run against a system interpreter whose site-packages would leak
# into the test environment.
SYS_EXEC="$("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"
if [[ "$SYS_EXEC" != "$PYTHON_BIN" ]]; then
  echo "error: sys.executable ($SYS_EXEC) != $PYTHON_BIN" >&2
  exit 1
fi

# Pin the interpreter major.minor so we never download wheels for another ABI.
PY_VERSION="$("$PYTHON_BIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
if [[ "$PY_VERSION" != "3.12" ]]; then
  echo "error: CI requires Python 3.12, found $PY_VERSION" >&2
  exit 1
fi

CONSTRAINTS="$REPO_ROOT/constraints-ci.txt"
if [[ ! -f "$CONSTRAINTS" ]]; then
  echo "error: missing $CONSTRAINTS" >&2
  exit 1
fi

# Keep the root constraints file and the CI copy in sync when both exist.
CI_CONSTRAINTS="$REPO_ROOT/scripts/ci/constraints-ci.txt"
if [[ -f "$CI_CONSTRAINTS" ]]; then
  if ! diff -u "$CONSTRAINTS" "$CI_CONSTRAINTS"; then
    echo "error: constraints-ci.txt copies diverged" >&2
    exit 1
  fi
fi

set -x
"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install --no-deps -c "$CONSTRAINTS" \
  pytest numpy torch
set +x

echo "test dependencies installed"
