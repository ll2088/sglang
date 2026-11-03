#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Install the minimal dependencies needed by the quant regression tests
# without resolving the full sglang dependency tree.
#
# Usage: scripts/ci/install_test_deps.sh
#
# The script intentionally uses the *current* interpreter (no .venv
# activation) so that it works both locally and in CI where the runner
# already provides a Python environment.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "error: ${PYTHON_BIN} not found" >&2
  exit 1
fi

echo "python: $(${PYTHON_BIN} -V 2>&1)"
echo "pip:    $(${PYTHON_BIN} -m pip -V 2>&1)"

CONSTRAINTS_FILE="${CONSTRAINTS_FILE:-scripts/ci/constraints-ci.txt}"

PIP_ARGS=(
  --no-deps
  --disable-pip-version-check
  --no-input
)

if [[ -f "${CONSTRAINTS_FILE}" ]]; then
  PIP_ARGS+=(-c "${CONSTRAINTS_FILE}")
  echo "using constraints: ${CONSTRAINTS_FILE}"
else
  echo "warning: constraints file ${CONSTRAINTS_FILE} not found; continuing without it" >&2
fi

# Install the project in editable mode without pulling transitive deps.
# The quant regression tests only need pytest plus the modules that are
# already importable in the CI image.
echo "running: ${PYTHON_BIN} -m pip install ${PIP_ARGS[*]} -e python"
"${PYTHON_BIN}" -m pip install "${PIP_ARGS[@]}" -e python

# Smoke import check: fail fast with a full traceback instead of a truncated
# log tail.
echo "running import smoke check"
"${PYTHON_BIN}" - <<'PY'
import traceback

MODULES = [
    "sglang.srt.layers.quantization.utils",
    "sglang.srt.layers.quantization.compressed_tensors.utils",
    "sglang.srt.layers.quantization.quark.utils",
]

failed = False
for name in MODULES:
    try:
        __import__(name)
    except Exception:
        failed = True
        print(f"import failed: {name}")
        traceback.print_exc()

if failed:
    raise SystemExit(1)
print("import smoke check passed")
PY

echo "done"
