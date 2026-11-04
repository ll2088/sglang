"""CPU-only test entry for the dense 2CTA MLA Q-head padding predicate.

This script deliberately imports nothing beyond the standard library. It loads
``mla_head_padding.py`` and ``test_mla_head_padding_standalone.py`` by file
path so the predicate boundary tests can run on a CPU-only runner without
installing the sglang dependency tree (torch, cutlass, quack, ...).

Usage::

    python python/sglang/kernels/ops/attention/flash_attn/cute/run_mla_head_padding_cpu_tests.py

Exit code 0 means every assertion passed.
"""

import importlib.util
import pathlib
import sys


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    cute_dir = pathlib.Path(__file__).resolve().parent
    predicate = _load_by_path(
        "_fa_cute_mla_head_padding_cpu_entry", cute_dir / "mla_head_padding.py"
    )
    standalone = _load_by_path(
        "_fa_cute_mla_head_padding_standalone_cpu_entry",
        cute_dir / "test_mla_head_padding_standalone.py",
    )

    pads = predicate.mla_2cta_pads_dense_qheads
    accepts = predicate.mla_2cta_accepts_dense_qheads

    # Boundary head counts for MQA (single KV head).
    assert pads(64, 1) is False
    assert pads(65, 1) is True
    assert pads(96, 1) is True
    assert pads(127, 1) is True
    assert pads(128, 1) is False
    assert pads(24, 1) is False
    assert pads(0, 1) is False
    assert pads(-1, 1) is False
    assert pads(129, 1) is False
    # Non-MQA never pads.
    assert pads(96, 2) is False
    assert pads(65, 2) is False
    assert pads(96, 0) is False
    # Boolean MQA flag form used by the usability check.
    assert pads(64, True) is False
    assert pads(65, True) is True
    assert pads(96, True) is True
    assert pads(127, True) is True
    assert pads(128, True) is False
    assert pads(0, True) is False
    assert pads(-1, True) is False
    assert pads(96, False) is False

    # Acceptance rule shared by the kernel and config validation.
    assert accepts(128, 1) is True
    assert accepts(64, 1) is True
    assert accepts(96, 1) is True
    assert accepts(65, 1) is True
    assert accepts(127, 1) is True
    assert accepts(24, 1) is False
    assert accepts(96, 2) is False
    assert accepts(0, 1) is False
    assert accepts(-1, 1) is False
    assert accepts(129, 1) is False

    # The standalone module must expose the same rule.
    assert standalone.mla_2cta_pads_dense_qheads is not None
    assert standalone.mla_2cta_accepts_dense_qheads is not None

    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
