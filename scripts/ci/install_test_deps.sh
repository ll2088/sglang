#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Install the minimal dependency set for the quantization regression tests
# without resolving the full sglang dependency tree.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONSTRAINTS="${SCRIPT_DIR}/constraints-ci.txt"

if [[ ! -f "${CONSTRAINTS}" ]]; then
  echo "error: constraints file not found: ${CONSTRAINTS}" >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" -m pip --version
"${PYTHON_BIN}" -m pip install --disable-pip-version-check --no-deps \
  -c "${CONSTRAINTS}" \
  -r "${CONSTRAINTS}"

# Smoke import check: fail fast with a clear message if the modules touched by
# the regression tests cannot be imported.
"${PYTHON_BIN}" - <<'PY'
import importlib
import sys

modules = [
    "sglang.srt.layers.quantization.utils",
    "sglang.srt.layers.quantization.compressed_tensors.utils",
    "sglang.srt.layers.quantization.quark.utils",
]

failed = []
for name in modules:
    try:
        importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001 - report all import failures
        failed.append((name, exc))

if failed:
    for name, exc in failed:
        print(f"import smoke check failed: {name}: {exc!r}", file=sys.stderr)
    sys.exit(1)

print("import smoke check passed")
PY

"${PYTHON_BIN}" -m pip freeze >&2 || true
