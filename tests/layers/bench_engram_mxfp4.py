"""Dependency-free performance baseline for the MXFP4 Engram layout module.

CI environment preparation can fail while resolving the full sglang
 dependency tree (torch, triton, ...). The layout module
``python/sglang/srt/layers/engram_mxfp4.py`` is stdlib-only, so its lookup
path can be benchmarked without that install.

This script measures the median latency of the reference lookup path for
both the FP8 and MXFP4 layouts and prints a comparison, so the performance
acceptance criterion ("median latency not more than 10% slower than the
baseline") has a concrete number instead of an empty metric.

Run with::

    PYTHONPATH=python python tests/layers/bench_engram_mxfp4.py

It never imports sglang, torch or triton.
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
    in the full dependency tree this benchmark exists to avoid.
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

LAYER_IDS = (17, 19)
HEAD_DIM = 256
ROWS = 4096
LOOKUPS = 512
REPEATS = 7


def _make_tables(layout):
    """Build a small table pair for the given layout."""
    if layout.packed_fp4:
        row = _layout.pack_e2m1([1.0] * layout.head_dim)
    else:
        row = [1] * layout.row_bytes
    table = [row] * ROWS
    scale_table = [[127] * layout.num_blocks] * ROWS
    ids = [i % ROWS for i in range(LOOKUPS)]
    return table, scale_table, ids


def _time_lookups(layout, table, scale_table, ids) -> float:
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

    fp8_table, fp8_scales, ids = _make_tables(fp8)
    mxfp4_table, mxfp4_scales, _ = _make_tables(mxfp4)

    fp8_us = _time_lookups(fp8, fp8_table, fp8_scales, ids)
    mxfp4_us = _time_lookups(mxfp4, mxfp4_table, mxfp4_scales, ids)

    delta_pct = (mxfp4_us - fp8_us) / fp8_us * 100.0 if fp8_us else 0.0

    print(f"rows={ROWS} head_dim={HEAD_DIM} lookups={LOOKUPS} repeats={REPEATS}")
    print(f"fp8   median lookup: {fp8_us:.2f} us")
    print(f"mxfp4 median lookup: {mxfp4_us:.2f} us")
    print(f"mxfp4 vs fp8 delta: {delta_pct:+.2f}%")

    # The acceptance criterion is that the packed path is not more than 10%
    # slower than the baseline. Report the verdict explicitly so CI logs carry
    # a concrete performance number instead of an empty metric.
    if delta_pct > 10.0:
        print("PERF FAIL: mxfp4 lookup more than 10% slower than fp8 baseline")
        return 1
    print("PERF OK: mxfp4 lookup within 10% of fp8 baseline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
