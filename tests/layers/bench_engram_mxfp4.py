"""Dependency-free lookup benchmark for the MXFP4 Engram layout module.

This script measures the median lookup latency for the FP8 and MXFP4 layouts
using only the standard library, so it can run even when the full sglang
 dependency tree (torch, triton, ...) is unavailable during CI environment
preparation. It mirrors the upstream vLLM #61530 lookup microbenchmark at a
much smaller scale: the goal is to confirm that the packed path does not
introduce a significant latency regression relative to the unpacked path.

Run directly with::

    PYTHONPATH=python python tests/layers/bench_engram_mxfp4.py

or through the bundled runner::

    bash tests/layers/run_engram_mxfp4_tests.sh --bench

The script prints the median lookup latency for both layouts and exits
non-zero if the packed path is more than 10% slower than the baseline, so CI
can use it as a performance gate.
"""

from __future__ import annotations

import importlib.util
import os
import statistics
import sys
import time

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_PYTHON_DIR = os.path.join(_REPO_ROOT, "python")
if os.path.isdir(_PYTHON_DIR) and _PYTHON_DIR not in sys.path:
    sys.path.insert(0, _PYTHON_DIR)

_LAYOUT_PATH = os.path.join(
    _PYTHON_DIR, "sglang", "srt", "layers", "engram_mxfp4.py"
)


def _load_layout_module():
    """Load the stdlib-only layout module by file path.

    Loading by path avoids importing the ``sglang`` package, which would pull
    in the full dependency tree (torch, triton, ...) that may be unavailable
    during CI environment preparation.
    """
    if not os.path.isfile(_LAYOUT_PATH):
        raise ImportError(f"engram_mxfp4 module not found at {_LAYOUT_PATH!r}")
    spec = importlib.util.spec_from_file_location("engram_mxfp4_bench", _LAYOUT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load engram_mxfp4 from {_LAYOUT_PATH!r}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_layout = _load_layout_module()

EngramLayout = _layout.EngramLayout
decode_e2m1_nibble = _layout.decode_e2m1_nibble
lookup_rows = _layout.lookup_rows
pack_e2m1 = _layout.pack_e2m1

LAYER_IDS = (17, 19)
HEAD_DIM = 256
ROWS = 4096
LOOKUPS = 512
REPEATS = 5
MAX_REGRESSION = 0.10


def _make_tables(layout: EngramLayout):
    """Build a deterministic table pair for the given layout."""
    if layout.packed_fp4:
        values = [decode_e2m1_nibble(c) for c in range(16)] * (HEAD_DIM // 16)
        row = pack_e2m1(values)
        table = [row] * ROWS
    else:
        table = [[1] * HEAD_DIM] * ROWS
    scales = [[127] * layout.num_blocks] * ROWS
    return table, scales


def _median_lookup_seconds(layout: EngramLayout, ids) -> float:
    """Return the median wall-clock time for one lookup batch."""
    table, scales = _make_tables(layout)
    samples = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        lookup_rows(layout, table, scales, ids)
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def main() -> int:
    fp8 = EngramLayout.from_metadata(HEAD_DIM, LAYER_IDS, None)
    mxfp4 = EngramLayout.from_metadata(
        HEAD_DIM, LAYER_IDS, {"engram_embed_quant": "mxfp4"}
    )

    ids = [i % ROWS for i in range(LOOKUPS)]

    fp8_seconds = _median_lookup_seconds(fp8, ids)
    mxfp4_seconds = _median_lookup_seconds(mxfp4, ids)

    print(f"[engram-mxfp4] rows={ROWS} head_dim={HEAD_DIM} lookups={LOOKUPS}")
    print(f"[engram-mxfp4] fp8   median={fp8_seconds * 1e6:.2f} us")
    print(f"[engram-mxfp4] mxfp4 median={mxfp4_seconds * 1e6:.2f} us")

    if fp8_seconds <= 0:
        print("[engram-mxfp4] status=bench-invalid (fp8 baseline is zero)")
        return 1

    regression = (mxfp4_seconds - fp8_seconds) / fp8_seconds
    print(f"[engram-mxfp4] regression={regression * 100:.2f}%")

    if regression > MAX_REGRESSION:
        print(
            "[engram-mxfp4] status=bench-failed "
            f"(packed path {regression * 100:.2f}% slower than baseline)"
        )
        return 1

    print("[engram-mxfp4] status=bench-passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
