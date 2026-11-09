#!/usr/bin/env bash
# Lightweight environment preparation entry for the run_suite verification
# dimensions. CI must call this script (or test/unit/setup_env.sh) instead of
# `pip install -e python`, which resolves the full sglang dependency set
# (torch, kernels, ...) and fails before any test can run.
#
# The guards below run before any pip invocation so an accidental editable
# install or a full-dependency resolution is rejected with a clear diagnostic
# (exit 2) instead of silently continuing.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[setup] using ${SCRIPT_DIR}/setup_env.sh"

# 1) Reject editable installs as early as possible: they resolve the full
#    sglang dependency set and are exactly what this entry replaces.
for arg in "$@"; do
  case "${arg}" in
    -e|--editable|-e=*|--editable=*)
      echo "[setup] ERROR: editable installs are not supported by the lightweight setup entry" >&2
      echo "[setup] use 'bash test/unit/setup_env.sh' (pytest only) instead of 'pip install -e python'" >&2
      exit 2
      ;;
  esac
done

# 2) Delegate to the pytest-only script next to the unit tests.
LIGHTWEIGHT="${SCRIPT_DIR}/test/unit/setup_env.sh"
if [[ ! -f "${LIGHTWEIGHT}" ]]; then
  echo "[setup] ERROR: missing lightweight setup script: ${LIGHTWEIGHT}" >&2
  exit 2
fi

bash "${LIGHTWEIGHT}" "$@"
