"""Host-side micro-benchmark for the BF16 capability negotiation helpers.

This is a pure-Python benchmark: it measures the cost of the dtype/alignment
predicate and the packed-bin extraction, which is what the dispatcher pays per
launch.  It skips cleanly when numpy is unavailable so CI can run it without a
GPU or the full runtime.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, relpath: str):
    path = _REPO_ROOT / relpath
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


it = _load_module(
    "_dsa_indexer_topk_helpers_bench",
    "python/sglang/srt/state_capturer/indexer_topk.py",
)


def _timeit(fn, iters: int = 20000) -> float:
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - start) / iters


def test_bf16_capability_predicate_is_cheap() -> None:
    """The negotiation predicate must stay far below a kernel launch."""
    per_call = _timeit(lambda: it.takes_bf16_logits_for(16, 16, 1, 65536))
    # 10 us is a generous ceiling; a launch is tens of microseconds.
    assert per_call < 10e-6, f"predicate too slow: {per_call * 1e6:.3f} us"


def test_bf16x2_extraction_matches_scalar() -> None:
    """Packed extraction must agree with the scalar reference on random bits."""
    rng = np.random.default_rng(0)
    words = rng.integers(0, 1 << 32, size=4096, dtype=np.uint64).astype(np.uint32)
    for w in words:
        lo, hi = it.host_convert_bf16x2_to_uint8x2(int(w))
        assert lo == it.host_convert_bf16_to_uint8(int(w) & 0xFFFF)
        assert hi == it.host_convert_bf16_to_uint8((int(w) >> 16) & 0xFFFF)
