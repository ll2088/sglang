#!/usr/bin/env bash
# Run the BF16 DSA top-k capability tests in a minimal venv that never
# installs the full sglang package (and therefore never resolves torch,
# py-spy, quack-kernels, pyzmq, ...).
#
# Usage: tests/kernels/run_bf16_minimal.sh [pytest args...]
set -euxo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# Pin the interpreter: the runner's default python may be newer than the
# wheels available for the minimal deps.
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "error: ${PYTHON_BIN} not found; set PYTHON_BIN to a 3.10-3.12 interpreter" >&2
  exit 1
fi

VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv-bf16-minimal}"
if [ ! -x "${VENV_DIR}/bin/python" ]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip
# Full, untruncated pip output on failure: no `| tail` here on purpose.
python -m pip install -r tests/kernels/requirements-minimal.txt

# Assert the capability module loads without pulling in torch or sglang.
python - <<'PY'
import sys
sys.path.insert(0, "python")
import sglang.kernels.dsa_topk_bf16_capability as cap
assert hasattr(cap, "takes_bf16_logits_for")
assert "torch" not in sys.modules, "torch must not be imported"
assert "sglang.srt" not in sys.modules, "sglang.srt must not be imported"
print("minimal-env capability import OK")
PY

python -m pytest -q tests/kernels/test_dsa_topk_bf16.py "$@"
