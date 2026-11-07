#!/usr/bin/env bash
# Create an isolated venv that installs ONLY the minimal test dependencies,
# then run the BF16 top-k capability tests.  This avoids `pip install -e .`
# / `pip install sglang`, which resolves the full runtime dependency set and
# is what previously failed during CI environment preparation.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-${HERE}/.venv-bf16-minimal}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip >/dev/null
python -m pip install -r "${HERE}/requirements-minimal.txt"

# Make the interpreter and installed set visible in CI logs.
python -c "import sys; print('python:', sys.executable)"
python -m pip list

cd "${HERE}/../.."
exec python -m pytest -q -p no:cacheprovider "${HERE}/test_dsa_topk_bf16_kernel.py" "$@"
