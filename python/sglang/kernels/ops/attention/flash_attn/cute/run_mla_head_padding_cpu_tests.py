"""CPU-only entry point for the dense 2CTA MLA Q-head padding rule.

This script deliberately imports nothing beyond the standard library and the
sibling ``mla_head_padding.py`` (loaded by file path), so it can run in a
minimal CI job without CUDA/CuTe, torch, quack or the sglang package. It
mirrors the boundary checks in ``test_mla_head_padding_standalone.py`` and the
pack_gqa fallback resolution in ``test_pack_gqa.py`` without importing either
of those modules (which may pull in CUDA deps at collection time).

Usage::

    python run_mla_head_padding_cpu_tests.py

Exit code 0 means every check passed.
"""

import importlib.util
import pathlib
import sys


def _load_predicate_module():
    path = pathlib.Path(__file__).with_name("mla_head_padding.py")
    spec = importlib.util.spec_from_file_location(
        "_fa_cute_mla_head_padding_cpu", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def main():
    module = _load_predicate_module()
    pads = module.mla_2cta_pads_dense_qheads
    accepts = module.mla_2cta_accepts_dense_qheads

    # Boundary head counts for MQA (single KV head).
    for nheads, expected in [
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
        _check(
            pads(nheads, 1) is expected,
            f"pads({nheads}, 1) != {expected}",
        )

    # Non-MQA never pads.
    for nheads in (65, 96, 127):
        _check(pads(nheads, 2) is False, f"pads({nheads}, 2) should be False")

    # Boolean MQA flag form used by the usability check.
    for nheads, expected in [
        (64, False),
        (65, True),
        (127, True),
        (128, False),
        (0, False),
        (-1, False),
    ]:
        _check(
            pads(nheads, True) is expected,
            f"pads({nheads}, True) != {expected}",
        )
    _check(pads(96, False) is False, "pads(96, False) should be False")

    # Acceptance rule: dividing heads accepted as-is, 65-127 MQA padded.
    for nheads, heads_kv, expected in [
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
        _check(
            accepts(nheads, heads_kv) is expected,
            f"accepts({nheads}, {heads_kv}) != {expected}",
        )

    # Padded rows keep the real head order and land past the valid heads.
    for nheads in (65, 72, 96, 127):
        _check(pads(nheads, 1), f"pads({nheads}, 1) should be True")
        rows = list(range(128))
        _check(rows[:nheads] == list(range(nheads)), "head order changed")
        _check(all(r >= nheads for r in rows[nheads:]), "padded rows overlap")

    # The predicate must resolve without the sglang package or CUDA deps.
    for name in list(sys.modules):
        if name == "sglang" or name.startswith("sglang."):
            del sys.modules[name]
    sys.modules["sglang"] = None
    sys.modules["cutlass"] = None
    sys.modules["torch"] = None
    sys.modules["quack"] = None
    try:
        module2 = _load_predicate_module()
        _check(module2.mla_2cta_pads_dense_qheads(96, 1) is True, "fallback failed")
    finally:
        for name in ("sglang", "cutlass", "torch", "quack"):
            sys.modules.pop(name, None)

    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
