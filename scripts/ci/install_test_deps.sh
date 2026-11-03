#!/usr/bin/env bash
# Install only the minimal dependencies required by the self-contained
# regression tests. Never install the sglang package itself here: doing so
# triggers resolution of the full runtime dependency tree (py-spy, pyzmq,
# quack-kernels, ...) which is unnecessary for these tests.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"

log() { printf '[install_test_deps] %s\n' "$*" >&2; }

log "python: $(${PYTHON_BIN} --version 2>&1)"
log "repo root: ${REPO_ROOT}"

# Refuse to run under an interpreter that has no matching wheels for the
# pinned test dependencies (e.g. cp314 during the migration window).
if ! ${PYTHON_BIN} - <<'PY'
import sys
if sys.version_info < (3, 9) or sys.version_info >= (3, 14):
    raise SystemExit(1)
PY
then
  log "unsupported Python version; expected >=3.9,<3.14"
  exit 1
fi

log "installing minimal test dependencies (no sglang, no deps)"
${PYTHON_BIN} -m pip install \
  --no-deps \
  --disable-pip-version-check \
  -r "${SCRIPT_DIR}/requirements-test-min.txt" \
  -c "${SCRIPT_DIR}/constraints-ci.txt"

log "verifying pytest import"
${PYTHON_BIN} -c "import pytest; print('pytest', pytest.__version__)"

log "done"
