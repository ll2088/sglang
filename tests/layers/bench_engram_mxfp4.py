"""Dependency-free benchmark for the MXFP4 Engram lookup path.

This script measures the median lookup latency for the FP8 and MXFP4
layouts using only the standard library, so it can run in the same minimal
CI environment as the layout tests (no torch, no triton, no sglang). It
prints the median latency for both layouts and fails if the packed path is
more than 10% slower than the FP8 baseline, matching the acceptance
criterion for the performance dimension.

Run with::

    PYTHONPATH=python python tests/layers/bench_engram_mxfp4.py

or via the bundled runner::

    bash tests/layers/run_engram_mxfp4_tests.sh --bench
"""

from __future__ import annotations

import importlib.util
import os
import statistics
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PYTHON_DIR = os.path.join(_REPO_ROOT, "python")
if os.path.isdir(_PYTHON_DIR) and _PYTHON_DIR not in sys.path:
    sys.path.insert(0, _PYTHON_DIR)

_LAYOUT_PATH = os.path.join(
    _PYTHON_DIR, "sglang", "srt", "layers", "engram_mxfp4.py"
)


def _load_layout():
    """Load the stdlib-only layout module by file path (never imports sglang)."""
    if not os.path.isfile(_LAYOUT_PATH):
        raise ImportError(f"engram_mxfp4 module not found at {_LAYOUT_PATH!r}")
    spec = importlib.util.spec_from_file_location("engram_mxfp4_bench", _LAYOUT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load engram_mxfp4 from {_LAYOUT_PATH!r}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_layout = _load_layout()

LAYER_IDS = (17, 19)
HEAD_DIM = 256
ROWS = 4096
LOOKUPS = 512
REPEATS = 5
MAX_SLOWDOWN = 1.10


def _make_tables(layout):
    """Build a small deterministic table for the given layout."""
    if layout.packed_fp4:
        row = _layout.pack_e2m1([1.0] * layout.head_dim)
    else:
        row = [1] * layout.head_dim
    scales = [127] * layout.num_blocks
    table = [row] * ROWS
    scale_table = [scales] * ROWS
    return table, scale_table


def _time_lookup(layout, table, scale_table, ids):
    """Return the median lookup latency in microseconds over REPEATS runs."""
    samples = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        _layout.lookup_rows(layout, table, scale_table, ids)
        samples.append((time.perf_counter() - start) * 1e6)
    return statistics.median(samples)


def main() -> int:
    fp8 = _layout.EngramLayout.from_metadata(HEAD_DIM, LAYER_IDS, None)
    mxfp4 = _layout.EngramLayout.from_metadata(
        HEAD_DIM, LAYER_IDS, {"engram_embed_quant": "mxfp4"}
    )

    ids = [i % ROWS for i in range(LOOKUPS)]
    fp8_table, fp8_scales = _make_tables(fp8)
    mxfp4_table, mxfp4_scales = _make_tables(mxfp4)

    # Warm up both paths so the first-call overhead does not skew the median.
    _layout.lookup_rows(fp8, fp8_table, fp8_scales, ids[:8])
    _layout.lookup_rows(mxfp4, mxfp4_table, mxfp4_scales, ids[:8])

    fp8_us = _time_lookup(fp8, fp8_table, fp8_scales, ids)
    mxfp4_us = _time_lookup(mxfp4, mxfp4_table, mxfp4_scales, ids)

    print(f"fp8   median lookup: {fp8_us:.2f} us")
    print(f"mxfp4 median lookup: {mxfp4_us:.2f} us")
    print(
        "row bytes: fp8="
        f"{fp8.total_row_bytes()} mxfp4={mxfp4.total_row_bytes()}"
    )

    if fp8_us <= 0:
        print("baseline latency is not measurable", file=sys.stderr)
        return 1
    slowdown = mxfp4_us / fp8_us
    print(f"mxfp4/fp8 ratio: {slowdown:.3f} (limit {MAX_SLOWDOWN:.2f})")
    if slowdown > MAX_SLOWDOWN:
        print(
            f"packed path is {slowdown:.3f}x the baseline, above the "
            f"{MAX_SLOWDOWN:.2f}x limit",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
