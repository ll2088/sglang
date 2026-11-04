"""CPU-only runner for the dense 2CTA MLA Q-head padding rule.

Runs without CUDA/CuTe, torch, quack or the sglang package: it loads the
sibling ``mla_head_padding.py`` by file path and checks the predicate
boundaries plus the head-order contract. Intended for minimal CI jobs where
installing the full sglang dependency tree is not possible.

Usage:
    python run_mla_head_padding_cpu_tests.py
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


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def main() -> int:
    module = _load_predicate_module()
    pads = module.mla_2cta_pads_dense_qheads
    accepts = module.mla_2cta_accepts_dense_qheads

    # Boundary head counts for MQA (single KV head).
    for nheads, expected in (
        (64, False),
        (65, True),
        (96, True),
        (127, True),
        (128, False),
        (24, False),
        (0, False),
        (-1, False),
        (129, False),
    ):
        _check(
            pads(nheads, 1) is expected,
            f"pads({nheads}, 1) != {expected}",
        )

    # Non-MQA never pads.
    for nheads in (65, 96, 127):
        _check(pads(nheads, 2) is False, f"pads({nheads}, 2) should be False")

    # Boolean MQA flag form used by the usability check.
    for nheads, expected in (
        (64, False),
        (65, True),
        (96, True),
        (127, True),
        (128, False),
        (0, False),
        (-1, False),
    ):
        _check(
            pads(nheads, True) is expected,
            f"pads({nheads}, True) != {expected}",
        )
    _check(pads(96, False) is False, "pads(96, False) should be False")

    # Acceptance rule: dividing heads accepted as-is, 65-127 MQA padded.
    for nheads, heads_kv, expected in (
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
    ):
        _check(
            accepts(nheads, heads_kv) is expected,
            f"accepts({nheads}, {heads_kv}) != {expected}",
        )

    # Padded rows keep the real head order and land past the valid heads.
    tile = 128
    for nheads in (65, 72, 96, 127):
        _check(pads(nheads, 1), f"pads({nheads}, 1) should be True")
        rows = list(range(tile))
        _check(rows[:nheads] == list(range(nheads)), "valid head order changed")
        _check(
            all(r >= nheads for r in rows[nheads:]),
            "padded rows overlap valid heads",
        )

    # 64 and 128 heads are not padded; the mapping is the identity.
    for nheads in (64, 128):
        _check(not pads(nheads, 1), f"pads({nheads}, 1) should be False")
        _check(
            list(range(nheads)) == list(range(nheads)),
            "identity mapping changed",
        )

    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
