#!/usr/bin/env bash
# Pure-stdlib runner for the MXFP4 Engram layout checks.
#
# CI environment preparation can fail while resolving the full sglang
# dependency tree (torch, triton, py-spy, pybase64, pydantic, quack-kernels,
# ...). The layout module python/sglang/srt/layers/engram_mxfp4.py is
# stdlib-only, so its checks must be runnable with no third-party dependency
# at all. This script is the dependency-free entry point used by the four CI
# dimensions (tests, stability, performance, usability) when pytest cannot be
# installed.
#
# It never installs anything and never imports the sglang package. It prints
# the executed check list and a final status line so CI logs show a decidable
# result instead of an environment-preparation failure.
#
# Usage (from the repository root):
#   bash tests/layers/run_engram_mxfp4_stdlib.sh
#   bash tests/layers/run_engram_mxfp4_stdlib.sh --repeat 3
#   bash tests/layers/run_engram_mxfp4_stdlib.sh --durations

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}python"

SELFCHECK="tests/layers/test_engram_mxfp4_selfcheck.py"
LAYOUT="python/sglang/srt/layers/engram_mxfp4.py"

REPEAT=1
SHOW_DURATIONS=0
while [ $# -gt 0 ]; do
    case "$1" in
        --repeat)
            REPEAT="${2:-1}"
            shift 2
            ;;
        --durations)
            SHOW_DURATIONS=1
            shift
            ;;
        *)
            echo "[engram-mxfp4] unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if [ ! -f "$SELFCHECK" ]; then
    echo "[engram-mxfp4] status=failed (missing $SELFCHECK)"
    exit 1
fi
if [ ! -f "$LAYOUT" ]; then
    echo "[engram-mxfp4] status=failed (missing $LAYOUT)"
    exit 1
fi

echo "[engram-mxfp4] status=stdlib-selfcheck"
echo "[engram-mxfp4] PYTHONPATH=$PYTHONPATH"
echo "[engram-mxfp4] layout module: $LAYOUT"
echo "[engram-mxfp4] self-check: $SELFCHECK"
echo "[engram-mxfp4] repeat=$REPEAT durations=$SHOW_DURATIONS"

# Usability: the layout module must import without the sglang package and
# expose its public interface.
python -c "
import importlib.util
spec = importlib.util.spec_from_file_location('engram_mxfp4', '$LAYOUT')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
for name in ('EngramLayout', 'decode_e2m1_nibble', 'allocate_row_buffer'):
    assert hasattr(module, name), name
layout = module.EngramLayout.from_metadata(
    256, (17, 19), {'engram_embed_quant': 'mxfp4'}
)
assert layout.row_bytes == 128
assert layout.total_row_bytes() == 136
print('[engram-mxfp4] usability=ok')
"
usability_status=$?
if [ "$usability_status" -ne 0 ]; then
    echo "[engram-mxfp4] status=failed (usability exit $usability_status)"
    exit 1
fi

# Tests / stability / performance: run the stdlib self-check, optionally
# repeated to surface state leakage or random failures, and optionally with
# per-case durations.
run_once() {
    local label="$1"
    if [ "$SHOW_DURATIONS" -eq 1 ]; then
        python - "$SELFCHECK" <<'PY'
import runpy
import sys
import time

path = sys.argv[1]
module = runpy.run_path(path)
tests = module['_collect_tests']()
start = time.perf_counter()
failures = []
for name, fn in tests:
    t0 = time.perf_counter()
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - report and continue
        failures.append((name, exc))
    dt = time.perf_counter() - t0
    print(f"[engram-mxfp4] duration {name} {dt:.6f}s")
total = time.perf_counter() - start
for name, exc in failures:
    print(f"[engram-mxfp4] FAIL {name}: {exc!r}")
print(f"[engram-mxfp4] {len(tests) - len(failures)}/{len(tests)} cases passed in {total:.6f}s")
sys.exit(1 if failures else 0)
PY
    else
        python "$SELFCHECK"
    fi
    local status=$?
    if [ "$status" -ne 0 ]; then
        echo "[engram-mxfp4] status=failed ($label exit $status)"
        exit 1
    fi
}

for i in $(seq 1 "$REPEAT"); do
    echo "[engram-mxfp4] run $i/$REPEAT"
    run_once "run $i"
done

echo "[engram-mxfp4] status=passed"
exit 0
