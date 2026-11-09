#!/usr/bin/env bash
# Root environment-preparation entry for the run_suite verification jobs.
#
# It delegates to test/unit/setup_env.sh, which installs only
# test/unit/requirements.txt (pytest). It must never resolve the full sglang
# dependency set: `pip install -e python` pulls torch/kernels/... and can fail
# before any test runs, which is exactly the failure mode this entry avoids.
set -euo pipefail

# Resolve paths from this script's own location, not the caller's CWD, so the
# delegation works no matter where CI invokes it from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT_SETUP="${SCRIPT_DIR}/test/unit/setup_env.sh"

if [[ ! -f "${UNIT_SETUP}" ]]; then
  echo "[setup] missing lightweight entry: ${UNIT_SETUP}" >&2
  exit 2
fi

# Reject any attempt to route this entry through an editable sglang install.
for arg in "$@"; do
  case "${arg}" in
    -e|--editable|-e=*|--editable=*)
      echo "[setup] refusing editable install (${arg}); use test/unit/setup_env.sh" >&2
      exit 2
      ;;
  esac
done

echo "[setup] delegating to ${UNIT_SETUP}"
exec bash "${UNIT_SETUP}" "$@"
