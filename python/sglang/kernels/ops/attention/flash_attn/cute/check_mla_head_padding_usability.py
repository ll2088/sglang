"""Usability check for the dense 2CTA MLA Q-head padding predicate.

Loads ``mla_head_padding.py`` by file path so the check does not import the
sglang package (and therefore does not pull in torch/quack/CUDA deps). This
mirrors the public query form used by callers::

    from ...pack_gqa import mla_2cta_pads_dense_qheads as f

Usage::

    python python/sglang/kernels/ops/attention/flash_attn/cute/check_mla_head_padding_usability.py
"""

import importlib.util
import pathlib
import sys


def _load_predicate():
    path = pathlib.Path(__file__).with_name("mla_head_padding.py")
    spec = importlib.util.spec_from_file_location(
        "_fa_cute_mla_head_padding_usability", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.mla_2cta_pads_dense_qheads


def main() -> int:
    f = _load_predicate()
    assert f(64, True) is False
    assert f(65, True) is True
    assert f(127, True) is True
    assert f(128, True) is False
    assert f(96, False) is False
    assert f(0, True) is False
    assert f(-1, True) is False
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
