#!/usr/bin/env python3
"""Stdlib-only runner for the dense 2CTA MLA Q-head padding tests.

This script deliberately imports nothing beyond the Python standard library
and the sibling ``mla_head_padding.py`` / ``test_mla_head_padding_standalone.py``
files, which are loaded by file path. It lets CI validate the padding predicate
without installing sglang, torch, quack, cutlass or CuTe.

Usage::

    python python/sglang/kernels/ops/attention/flash_attn/cute/run_mla_head_padding_cpu_tests.py

Exit code 0 means every check passed; 1 means at least one check failed.
"""

import importlib.util
import pathlib
import sys
import traceback


_HERE = pathlib.Path(__file__).resolve().parent


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_predicate_module():
    return _load_module("_fa_cute_mla_head_padding_runner", _HERE / "mla_head_padding.py")


def _load_standalone_tests():
    return _load_module(
        "_fa_cute_mla_head_padding_standalone_runner",
        _HERE / "test_mla_head_padding_standalone.py",
    )


def _check(condition, message):
    if not condition:
        raise AssertionError(message)


def _run_predicate_checks(predicate):
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
        _check(rows[:nheads] == list(range(nheads)), f"{nheads} heads lost identity order")
        _check(
            all(r >= nheads for r in rows[nheads:]),
            f"{nheads} heads: padded rows overlap valid heads",
        )
    for nheads in (64, 128):
        _check(not predicate(nheads, 1), f"{nheads} heads should not be padded")
        _check(list(range(nheads)) == list(range(nheads)), f"{nheads} heads mapping changed")


def _run_config_missing_checks():
    """The predicate must resolve even when flash_attn.cute.config is absent."""
    saved = {}
    for name in (
        "flash_attn",
        "flash_attn.cute",
        "flash_attn.cute.config",
        "sglang",
        "sglang.kernels",
        "sglang.kernels.ops",
        "sglang.kernels.ops.attention",
        "sglang.kernels.ops.attention.flash_attn",
        "sglang.kernels.ops.attention.flash_attn.cute",
        "sglang.kernels.ops.attention.flash_attn.cute.mla_head_padding",
    ):
        saved[name] = sys.modules.get(name, _MISSING)
        sys.modules[name] = None
    try:
        module = _load_predicate_module()
        predicate = module.mla_2cta_pads_dense_qheads
        _check(predicate(96, 1) is True, "predicate(96, 1) should be True without config")
        _check(predicate(64, 1) is False, "predicate(64, 1) should be False without config")
    finally:
        for name, value in saved.items():
            if value is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


class _Missing:
    pass


_MISSING = _Missing()


def main():
    failures = []

    def run(name, fn):
        try:
            fn()
        except Exception:
            failures.append(name)
            print(f"FAIL {name}")
            traceback.print_exc()
        else:
            print(f"ok   {name}")

    module = _load_predicate_module()
    predicate = module.mla_2cta_pads_dense_qheads
    accepts = module.mla_2cta_accepts_dense_qheads

    run("predicate boundaries", lambda: _run_predicate_checks(predicate))
    run("accepts boundaries", lambda: _run_accepts_checks(accepts))
    run("head order", lambda: _run_head_order_checks(predicate))
    run("config module missing", _run_config_missing_checks)

    def run_standalone():
        standalone = _load_standalone_tests()
        _check(
            standalone.mla_2cta_pads_dense_qheads(65, True) is True,
            "standalone predicate(65, True) should be True",
        )
        _check(
            standalone.mla_2cta_pads_dense_qheads(64, True) is False,
            "standalone predicate(64, True) should be False",
        )

    run("standalone module loads", run_standalone)

    if failures:
        print(f"\n{len(failures)} check(s) failed: {', '.join(failures)}")
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
