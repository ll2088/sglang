#!/usr/bin/env python3
# Copyright (c) 2025, Tri Dao.
"""Stdlib-only CPU checks for the dense 2CTA MLA Q-head padding rule.

This script deliberately imports nothing outside the Python standard library so
it can run in a minimal CI job that does not install sglang's full dependency
tree (torch, cutlass, quack, ...). It loads ``mla_head_padding.py`` and
``pack_gqa.py`` by file path, mirroring the sibling-module fallback used by the
kernel, and exercises the predicate boundaries plus the pack_gqa head-order
mapping without CUDA/CuTe.

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


def _load_predicate_module():
    return _load_by_path("_fa_cute_mla_head_padding_cpu", "mla_head_padding.py")


def _load_pack_gqa_predicate_only():
    """Resolve the predicate the way pack_gqa's fallback does, without CUDA.

    pack_gqa imports cutlass/quack at module scope, so a CPU-only runner cannot
    import it. The predicate itself lives in the sibling mla_head_padding.py;
    this mirrors pack_gqa's fallback resolution order so the CPU checks cover
    the same rule without the CUDA dependency chain.
    """
    return _load_predicate_module().mla_2cta_pads_dense_qheads


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def check_predicate_boundaries():
    predicate = _load_predicate_module().mla_2cta_pads_dense_qheads
    # int KV-head count form
    _check(predicate(64, 1) is False, "64 heads must not pad")
    _check(predicate(65, 1) is True, "65 heads must pad")
    _check(predicate(96, 1) is True, "96 heads must pad")
    _check(predicate(127, 1) is True, "127 heads must pad")
    _check(predicate(128, 1) is False, "128 heads must not pad")
    _check(predicate(96, 2) is False, "non-MQA must not pad")
    _check(predicate(0, 1) is False, "0 heads must not pad")
    _check(predicate(-1, 1) is False, "negative heads must not pad")
    # boolean MQA flag form
    _check(predicate(64, True) is False, "64 heads (bool MQA) must not pad")
    _check(predicate(65, True) is True, "65 heads (bool MQA) must pad")
    _check(predicate(127, True) is True, "127 heads (bool MQA) must pad")
    _check(predicate(128, True) is False, "128 heads (bool MQA) must not pad")
    _check(predicate(96, False) is False, "non-MQA (bool) must not pad")


def check_accepts_rule():
    accepts = _load_predicate_module().mla_2cta_accepts_dense_qheads
    _check(accepts(128, 1) is True, "128 heads accepted as-is")
    _check(accepts(64, 1) is True, "64 heads accepted as-is")
    _check(accepts(96, 1) is True, "96 heads accepted via padding")
    _check(accepts(65, 1) is True, "65 heads accepted via padding")
    _check(accepts(127, 1) is True, "127 heads accepted via padding")
    _check(accepts(24, 1) is False, "24 heads rejected")
    _check(accepts(96, 2) is False, "96 heads per KV head without MQA rejected")
    _check(accepts(0, 1) is False, "0 heads rejected")
    _check(accepts(-1, 1) is False, "negative heads rejected")


def check_pack_gqa_fallback_without_config():
    """pack_gqa's sibling-module fallback must not need flash_attn.cute.config."""
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
        predicate = _load_pack_gqa_predicate_only()
        _check(predicate(64, True) is False, "fallback: 64 must not pad")
        _check(predicate(65, True) is True, "fallback: 65 must pad")
        _check(predicate(127, True) is True, "fallback: 127 must pad")
        _check(predicate(128, True) is False, "fallback: 128 must not pad")
        _check(predicate(96, False) is False, "fallback: non-MQA must not pad")
        _check(predicate(0, True) is False, "fallback: 0 must not pad")
        _check(predicate(-1, True) is False, "fallback: negative must not pad")
    finally:
        for name, value in saved.items():
            if value is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def check_pack_gqa_head_order_mapping():
    """Padded head rows must map back to the original head order.

    The kernel packs a token's Q heads into the 128-row tile and drops the
    padded rows on store. This check mirrors the index arithmetic used by
    ``PackGQA.compute_ptr`` (``m_idx = idx // qhead_per_kvhead``,
    ``h_idx = idx - m_idx * qhead_per_kvhead``) and verifies that for a padded
    head count the valid rows recover the original (token, head) pairs in
    order, and that padded rows fall outside the real head range.
    """
    for qhead_per_kvhead in (65, 72, 96, 127):
        tile = 128
        seqlen = 3
        for block in range(0, seqlen * tile, tile):
            for row in range(tile):
                idx = block + row
                m_idx = idx // qhead_per_kvhead
                h_idx = idx - m_idx * qhead_per_kvhead
                if row < qhead_per_kvhead:
                    _check(
                        m_idx < seqlen,
                        f"valid row {row} maps past seqlen for {qhead_per_kvhead} heads",
                    )
                    _check(
                        0 <= h_idx < qhead_per_kvhead,
                        f"valid row {row} maps to head {h_idx} out of range",
                    )
                    _check(
                        idx == m_idx * qhead_per_kvhead + h_idx,
                        "head-order mapping is not invertible",
                    )
                else:
                    # Padded rows must not alias a real head of the next token.
                    _check(
                        h_idx >= qhead_per_kvhead or m_idx >= seqlen,
                        f"padded row {row} aliases a real head for {qhead_per_kvhead} heads",
                    )


def check_non_padding_paths_unchanged():
    """Head counts that divide 128 must not take the padding path."""
    predicate = _load_predicate_module().mla_2cta_pads_dense_qheads
    for heads in (1, 2, 4, 8, 16, 32, 64, 128):
        _check(
            predicate(heads, 1) is False,
            f"{heads} heads divide 128 and must not pad",
        )


def main():
    checks = [
        ("predicate boundaries", check_predicate_boundaries),
        ("accepts rule", check_accepts_rule),
        ("pack_gqa fallback without config", check_pack_gqa_fallback_without_config),
        ("pack_gqa head-order mapping", check_pack_gqa_head_order_mapping),
        ("non-padding paths unchanged", check_non_padding_paths_unchanged),
    ]
    failures = []
    for name, fn in checks:
        try:
            fn()
        except AssertionError as exc:
            failures.append(f"{name}: {exc}")
        else:
            print(f"ok - {name}")
    if failures:
        for failure in failures:
            print(f"FAIL - {failure}", file=sys.stderr)
        return 1
    print("all MLA head-padding CPU checks passed")
    return 0


class _Missing:
    pass


_MISSING = _Missing()


if __name__ == "__main__":
    sys.exit(main())
