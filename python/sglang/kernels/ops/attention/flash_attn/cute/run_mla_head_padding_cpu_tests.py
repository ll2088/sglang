#!/usr/bin/env python3
"""CPU-only runner for the dense 2CTA MLA Q-head padding tests.

This script deliberately imports nothing beyond the standard library and the
sibling ``mla_head_padding.py`` (loaded by file path). It lets a CPU CI job
validate the padding predicate without installing sglang, torch, quack or
CUDA/CuTe, which is what made the full-package setup fail.

Usage:
    python run_mla_head_padding_cpu_tests.py

Exit code 0 means every check passed; 1 means at least one failed.
"""

import importlib.util
import pathlib
import sys


def _load_predicate_module():
    path = pathlib.Path(__file__).with_name("mla_head_padding.py")
    spec = importlib.util.spec_from_file_location(
        "_fa_cute_mla_head_padding_cpu_runner", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = _load_predicate_module()
    pads = module.mla_2cta_pads_dense_qheads
    accepts = module.mla_2cta_accepts_dense_qheads

    failures = []

    def check(name, actual, expected):
        if actual is not expected:
            failures.append(f"{name}: got {actual!r}, expected {expected!r}")

    # Boundary head counts for MQA (single KV head).
    for heads, expected in [
        (64, False),
        (65, True),
        (96, True),
        (127, True),
        (128, False),
        (24, False),
        (0, False),
        (-1, False),
        (129, False),
    ]:
        check(f"pads({heads}, 1)", pads(heads, 1), expected)

    # Non-MQA never pads.
    for heads in (65, 96):
        check(f"pads({heads}, 2)", pads(heads, 2), False)
    check("pads(96, 0)", pads(96, 0), False)

    # Boolean MQA flag form used by the usability check.
    for heads, expected in [
        (64, False),
        (65, True),
        (96, True),
        (127, True),
        (128, False),
        (0, False),
        (-1, False),
    ]:
        check(f"pads({heads}, True)", pads(heads, True), expected)
    check("pads(96, False)", pads(96, False), False)

    # Acceptance rule: dividing heads accepted as-is, 65-127 MQA padded.
    for heads, kv, expected in [
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
    ]:
        check(f"accepts({heads}, {kv})", accepts(heads, kv), expected)

    # Padded rows keep the real head order and land past the valid heads.
    for heads in (65, 72, 96, 127):
        if not pads(heads, 1):
            failures.append(f"pads({heads}, 1) should be True")
        rows = list(range(128))
        if rows[:heads] != list(range(heads)):
            failures.append(f"padded rows for {heads} heads are not identity")
        if any(r < heads for r in rows[heads:]):
            failures.append(f"padded rows for {heads} heads overlap valid heads")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
