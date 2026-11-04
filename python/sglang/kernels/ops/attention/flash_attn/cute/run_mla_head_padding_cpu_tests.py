# Copyright (c) 2025, Tri Dao.
"""Run the stdlib-only MLA head padding tests without sglang/CUDA deps.

Usage::

    python python/sglang/kernels/ops/attention/flash_attn/cute/run_mla_head_padding_cpu_tests.py

This script only needs the standard library plus pytest. It loads the sibling
``mla_head_padding.py`` by file path, so it works in a checkout where the
sglang package (and its heavy dependency tree) is not installed.
"""

import importlib.util
import pathlib
import sys


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    here = pathlib.Path(__file__).resolve().parent
    rule = _load_module("_fa_cute_mla_head_padding_run", here / "mla_head_padding.py")
    pads = rule.mla_2cta_pads_dense_qheads
    accepts = rule.mla_2cta_accepts_dense_qheads

    checks = [
        (pads(64, True), False, "pads(64, True)"),
        (pads(65, True), True, "pads(65, True)"),
        (pads(127, True), True, "pads(127, True)"),
        (pads(128, True), False, "pads(128, True)"),
        (pads(96, False), False, "pads(96, False)"),
        (pads(0, True), False, "pads(0, True)"),
        (pads(-1, True), False, "pads(-1, True)"),
        (pads(96, 1), True, "pads(96, 1)"),
        (pads(96, 2), False, "pads(96, 2)"),
        (accepts(128, 1), True, "accepts(128, 1)"),
        (accepts(64, 1), True, "accepts(64, 1)"),
        (accepts(96, 1), True, "accepts(96, 1)"),
        (accepts(24, 1), False, "accepts(24, 1)"),
        (accepts(96, 2), False, "accepts(96, 2)"),
        (accepts(0, 1), False, "accepts(0, 1)"),
        (accepts(-1, 1), False, "accepts(-1, 1)"),
    ]
    failures = []
    for got, want, label in checks:
        if got is not want:
            failures.append(f"{label}: got {got!r}, want {want!r}")
    if failures:
        for line in failures:
            print(line, file=sys.stderr)
        return 1
    print(f"ok: {len(checks)} MLA head padding checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
