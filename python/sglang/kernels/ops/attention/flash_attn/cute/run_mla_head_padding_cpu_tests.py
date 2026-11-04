"""CPU-only runner for the dense 2CTA MLA Q-head padding predicate.

This script deliberately imports nothing beyond the standard library and the
sibling ``mla_head_padding.py`` (loaded by file path), so it can run on a
CPU-only runner without CUDA/CuTe, torch, quack or the sglang package. It is
intended for CI jobs that cannot install the full sglang dependency tree.

Usage::

    python python/sglang/kernels/ops/attention/flash_attn/cute/run_mla_head_padding_cpu_tests.py
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
    _check(pads(64, 1) is False, "64 heads must not pad")
    _check(pads(65, 1) is True, "65 heads must pad")
    _check(pads(96, 1) is True, "96 heads must pad")
    _check(pads(127, 1) is True, "127 heads must pad")
    _check(pads(128, 1) is False, "128 heads must not pad")

    # Too few heads to pad: still rejected.
    _check(pads(24, 1) is False, "24 heads must not pad")
    _check(pads(0, 1) is False, "0 heads must not pad")
    _check(pads(-1, 1) is False, "-1 heads must not pad")
    _check(pads(129, 1) is False, "129 heads must not pad")

    # Non-MQA (more than one KV head) never pads.
    _check(pads(96, 2) is False, "96 heads with 2 KV heads must not pad")
    _check(pads(65, 2) is False, "65 heads with 2 KV heads must not pad")
    _check(pads(96, 0) is False, "96 heads with 0 KV heads must not pad")

    # Boolean MQA flag form used by the usability check.
    _check(pads(64, True) is False, "64 heads with MQA flag must not pad")
    _check(pads(65, True) is True, "65 heads with MQA flag must pad")
    _check(pads(96, True) is True, "96 heads with MQA flag must pad")
    _check(pads(127, True) is True, "127 heads with MQA flag must pad")
    _check(pads(128, True) is False, "128 heads with MQA flag must not pad")
    _check(pads(0, True) is False, "0 heads with MQA flag must not pad")
    _check(pads(-1, True) is False, "-1 heads with MQA flag must not pad")
    _check(pads(96, False) is False, "96 heads without MQA flag must not pad")

    # Acceptance rule: dividing heads accepted as-is, 65-127 MQA padded.
    _check(accepts(128, 1) is True, "128 heads must be accepted")
    _check(accepts(64, 1) is True, "64 heads must be accepted")
    _check(accepts(96, 1) is True, "96 heads must be accepted")
    _check(accepts(65, 1) is True, "65 heads must be accepted")
    _check(accepts(127, 1) is True, "127 heads must be accepted")
    _check(accepts(24, 1) is False, "24 heads must be rejected")
    _check(accepts(96, 2) is False, "96 heads with 2 KV heads must be rejected")
    _check(accepts(0, 1) is False, "0 heads must be rejected")
    _check(accepts(-1, 1) is False, "-1 heads must be rejected")
    _check(accepts(129, 1) is False, "129 heads must be rejected")

    # Padded rows keep the real head order and land past the valid heads.
    for nheads in (65, 72, 96, 127):
        _check(pads(nheads, 1), f"{nheads} heads must pad")
        rows = list(range(128))
        _check(rows[:nheads] == list(range(nheads)), "valid rows must be identity")
        _check(
            all(r >= nheads for r in rows[nheads:]),
            "padded rows must land past the valid heads",
        )

    # 64 and 128 heads are not padded; the mapping is the identity.
    for nheads in (64, 128):
        _check(not pads(nheads, 1), f"{nheads} heads must not pad")
        _check(list(range(nheads)) == list(range(nheads)), "identity mapping")

    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
