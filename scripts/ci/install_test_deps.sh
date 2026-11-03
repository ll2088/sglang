#!/usr/bin/env bash
# Install the minimal dependency set required by the regression tests.
#
# The goal is to avoid resolving sglang's full runtime dependency graph
# (py-spy, quack-kernels, pyzmq, ...) during CI setup.  We install the
# project in editable mode with --no-deps and then install only the
# packages listed in constraints-ci.txt.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONSTRAINTS="${REPO_ROOT}/constraints-ci.txt"

if [[ ! -f "${CONSTRAINTS}" ]]; then
  echo "error: constraints file not found: ${CONSTRAINTS}" >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python}"

log() { printf '[install_test_deps] %s\n' "$*" >&2; }

log "python: $(${PYTHON_BIN} --version 2>&1)"
log "constraints: ${CONSTRAINTS}"

# 1. Install the project itself without pulling runtime dependencies.
log "installing sglang (editable, --no-deps)"
if ! "${PYTHON_BIN}" -m pip install -e "${REPO_ROOT}/python" --no-deps; then
  log "failed to install sglang in editable mode"
  exit 1
fi

# 2. Install only the constrained test dependencies.
log "installing test dependencies from constraints"
if ! "${PYTHON_BIN}" -m pip install -c "${CONSTRAINTS}" \
    pytest torch numpy transformers huggingface_hub compressed-tensors; then
  log "failed to install test dependencies"
  exit 1
fi

log "done"
