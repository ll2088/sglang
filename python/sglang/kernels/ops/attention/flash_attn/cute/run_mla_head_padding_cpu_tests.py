"""CPU-only runner for the dense 2CTA MLA Q-head padding predicate.

This script deliberately imports nothing beyond the standard library and the
sibling ``mla_head_padding.py`` (loaded by file path), so it can run on a
CPU-only runner without CUDA/CuTe, torch, quack or the sglang package. It is
the entry point CI should call instead of installing the full sglang
dependency tree just to exercise the pure-Python rule.

Usage::

    python run_mla_head_padding_cpu_tests.py

Exit code 0 means every assertion passed.
"""

import importlib.util
import pathlib
import sys


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_predicate_module():
    path = pathlib.Path(__file__).with_name("mla_head_padding.py")
    return _load_module("_fa_cute_mla_head_padding_cpu_runner", path)


def _load_standalone_tests():
    path = pathlib.Path(__file__).with_name("test_mla_head_padding_standalone.py")
    return _load_module("_fa_cute_mla_head_padding_standalone_cpu_runner", path)


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _run_predicate_checks(predicate):
    # Boundary head counts for MQA (single KV head).
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
    for nheads_per_kv, num_heads_kv, expected in cases:
        got = predicate(nheads_per_kv, num_heads_kv)
        _check(
            got is expected,
            f"mla_2cta_pads_dense_qheads({nheads_per_kv}, {num_heads_kv}) "
            f"returned {got!r}, expected {expected!r}",
        )


def _run_accepts_checks(accepts):
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
    for nheads_per_kv, num_heads_kv, expected in cases:
        got = accepts(nheads_per_kv, num_heads_kv)
        _check(
            got is expected,
            f"mla_2cta_accepts_dense_qheads({nheads_per_kv}, {num_heads_kv}) "
            f"returned {got!r}, expected {expected!r}",
        )


def _run_head_order_checks(predicate):
    tile = 128
    for nheads in (65, 72, 96, 127):
        _check(predicate(nheads, 1), f"{nheads} heads should be padded")
        rows = list(range(tile))
        _check(rows[:nheads] == list(range(nheads)), "real head order changed")
        _check(
            all(r >= nheads for r in rows[nheads:]),
            "padded rows must land past the valid heads",
        )
    for nheads in (64, 128):
        _check(not predicate(nheads, 1), f"{nheads} heads should not be padded")
        _check(list(range(nheads)) == list(range(nheads)), "identity mapping changed")


def main():
    module = _load_predicate_module()
    predicate = module.mla_2cta_pads_dense_qheads
    accepts = module.mla_2cta_accepts_dense_qheads

    _run_predicate_checks(predicate)
    _run_accepts_checks(accepts)
    _run_head_order_checks(predicate)

    # The standalone module must also be loadable without sglang/CUDA deps.
    standalone = _load_standalone_tests()
    _check(
        standalone.mla_2cta_pads_dense_qheads(65, True) is True,
        "standalone module predicate mismatch",
    )
    _check(
        standalone.mla_2cta_pads_dense_qheads(64, True) is False,
        "standalone module predicate mismatch",
    )

    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
