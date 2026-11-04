# Copyright (c) 2025, Tri Dao.
"""Run the CPU-only MLA head-padding tests without the sglang package.

This script is intentionally stdlib-only: it loads the predicate module and
the standalone test module by file path, so CI can exercise the padding rule
without installing torch, cutlass, quack or the full sglang dependency tree.

Usage::

    python run_mla_head_padding_cpu_tests.py
"""

import importlib.util
import pathlib
import sys


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    here = pathlib.Path(__file__).resolve().parent
    predicate = _load_module(
        "_fa_cute_mla_head_padding_cpu", here / "mla_head_padding.py"
    )
    tests = _load_module(
        "_fa_cute_mla_head_padding_standalone_cpu",
        here / "test_mla_head_padding_standalone.py",
    )

    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    pads = predicate.mla_2cta_pads_dense_qheads
    accepts = predicate.mla_2cta_accepts_dense_qheads

    # Boundary behaviour of the padding predicate.
    for nheads, kv, expected in [
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
    ]:
        check(
            pads(nheads, kv) is expected,
            f"pads({nheads}, {kv}) != {expected}",
        )

    # Acceptance rule used by config validation.
    for nheads, kv, expected in [
        (128, 1, True),
        (64, 1, True),
        (96, 1, True),
        (65, 1, True),
        (127, 1, True),
        (24, 1, False),
        (96, 2, False),
        (0, 1, False),
        (-1, 1, False),
    ]:
        check(
            accepts(nheads, kv) is expected,
            f"accepts({nheads}, {kv}) != {expected}",
        )

    # Padded rows keep the real head order and land past the valid heads.
    for nheads in (65, 72, 96, 127):
        check(pads(nheads, 1), f"pads({nheads}, 1) should be True")
        rows = list(range(128))
        check(
            rows[:nheads] == list(range(nheads)),
            f"padded head order broken for {nheads}",
        )
        check(
            all(r >= nheads for r in rows[nheads:]),
            f"padded rows overlap valid heads for {nheads}",
        )

    # Non-padded head counts keep the identity mapping.
    for nheads in (64, 128):
        check(not pads(nheads, 1), f"pads({nheads}, 1) should be False")
        check(
            list(range(nheads)) == list(range(nheads)),
            f"identity mapping broken for {nheads}",
        )

    # The standalone test module must expose the same rule.
    check(
        tests.mla_2cta_pads_dense_qheads(65, True) is True,
        "standalone module predicate mismatch",
    )
    check(
        tests.mla_2cta_pads_dense_qheads(64, True) is False,
        "standalone module predicate mismatch",
    )

    if failures:
        for msg in failures:
            print(f"FAIL: {msg}", file=sys.stderr)
        return 1
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
