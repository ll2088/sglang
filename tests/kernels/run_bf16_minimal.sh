#!/usr/bin/env bash
# Run the DSA BF16 capability tests in an isolated venv that never resolves
# sglang's full dependency set.  CI invokes this script instead of
# `pip install -e python`; it is also the reproducible local command.
#
# Usage: tests/kernels/run_bf16_minimal.sh [pytest args...]
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${here}/../.." && pwd)"
venv="${DSA_BF16_VENV:-${repo_root}/.venv-dsa-bf16}"

# Pin the interpreter: the full-deps path pulled cp314 and cp312 wheels at
# once, which is what broke resolution.  numpy 1.26 has no cp314 wheel.
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "error: ${PYTHON_BIN} not found; set PYTHON_BIN to a 3.12 interpreter" >&2
  exit 1
fi

if [ ! -x "${venv}/bin/python" ]; then
  "${PYTHON_BIN}" -m venv "${venv}"
fi

"${venv}/bin/python" -m pip install --disable-pip-version-check --no-input \
  -r "${here}/requirements-minimal.txt"

cd "${here}"
exec "${venv}/bin/python" -m pytest -c "${here}/pytest.ini" "$@"
