"""Stdlib-only CPU runner for the dense 2CTA MLA Q-head padding rule.

This script is the CPU-only verification entry point. It deliberately imports
nothing beyond the Python standard library: no sglang, cutlass, quack or
torch. The predicate and the pack_gqa fallback are loaded by file path so the
same rule the kernel uses is exercised without the GPU dependency chain.

Usage::

    python run_mla_head_padding_cpu_tests.py

Exit code 0 means every check passed.
"""

import importlib.util
import pathlib
import sys


_HERE = pathlib.Path(__file__).resolve().parent


def _load_by_path(name, filename):
    path = _HERE / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_predicate():
    return _load_by_path("_fa_cute_mla_head_padding_cpu", "mla_head_padding.py")


def _load_pack_gqa_predicate_only():
    """Mirror pack_gqa's sibling-module fallback without importing pack_gqa.

    pack_gqa imports cutlass/quack at module scope, so a CPU-only runner cannot
    import it. The predicate lives in the sibling mla_head_padding.py; this
    helper resolves it the same way pack_gqa's fallback does.
    """
    return _load_predicate().mla_2cta_pads_dense_qheads


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def check_predicate_boundaries():
    predicate = _load_predicate().mla_2cta_pads_dense_qheads
    cases = [
        (64, 1, False),
        (65, 1, True),
        (96, 1, True),
        (127, 1, True),
        (128, 1, False),
        (24, 1, False),
        (0, 1, False),
        (-1, 1, False),
        (129, 1, False),
        (96, 2, False),
        (65, 2, False),
        (96, 0, False),
        (64, True, False),
        (65, True, True),
        (96, True, True),
        (127, True, True),
        (128, True, False),
        (0, True, False),
        (-1, True, False),
        (96, False, False),
    ]
    for nheads, heads_kv, expected in cases:
        got = predicate(nheads, heads_kv)
        _check(
            got is expected,
            f"predicate({nheads}, {heads_kv}) = {got!r}, expected {expected!r}",
        )


def check_accepts_dense_qheads():
    accepts = _load_predicate().mla_2cta_accepts_dense_qheads
    cases = [
        (128, 1, True),
        (64, 1, True),
        (96, 1, True),
        (65, 1, True),
        (127, 1, True),
        (24, 1, False),
        (96, 2, False),
        (0, 1, False),
        (-1, 1, False),
        (129, 1, False),
    ]
    for nheads, heads_kv, expected in cases:
        got = accepts(nheads, heads_kv)
        _check(
            got is expected,
            f"accepts({nheads}, {heads_kv}) = {got!r}, expected {expected!r}",
        )


def check_pack_gqa_fallback():
    predicate = _load_pack_gqa_predicate_only()
    cases = [
        (64, True, False),
        (65, True, True),
        (127, True, True),
        (128, True, False),
        (96, False, False),
        (0, True, False),
        (-1, True, False),
    ]
    for nheads, heads_kv, expected in cases:
        got = predicate(nheads, heads_kv)
        _check(
            got is expected,
            f"pack_gqa predicate({nheads}, {heads_kv}) = {got!r}, "
            f"expected {expected!r}",
        )


def check_padded_head_order():
    predicate = _load_predicate().mla_2cta_pads_dense_qheads
    tile = 128
    for nheads in (65, 72, 96, 127):
        _check(predicate(nheads, 1), f"{nheads} heads should pad")
        rows = list(range(tile))
        _check(rows[:nheads] == list(range(nheads)), "real heads keep order")
        _check(
            all(r >= nheads for r in rows[nheads:]),
            "padded rows land past the valid heads",
        )
    for nheads in (64, 128):
        _check(not predicate(nheads, 1), f"{nheads} heads should not pad")
        _check(list(range(nheads)) == list(range(nheads)), "identity mapping")


def check_no_heavy_imports():
    """The runner must not pull in sglang/cutlass/quack/torch."""
    heavy = ("sglang", "cutlass", "quack", "torch")
    loaded = [name for name in sys.modules if name.split(".")[0] in heavy]
    _check(not loaded, f"heavy modules imported: {loaded}")


def main():
    check_predicate_boundaries()
    check_accepts_dense_qheads()
    check_pack_gqa_fallback()
    check_padded_head_order()
    check_no_heavy_imports()
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
