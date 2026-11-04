"""CPU-only usability check for the dense 2CTA MLA Q-head padding rule.

Verifies that the predicate can be resolved and queried without importing
the sglang package or CUDA/CuTe, mirroring the public query used by the
kernel and config validation. Intended for minimal CI jobs.

Usage:
    python check_mla_head_padding_usability.py
"""

import importlib.util
import pathlib
import sys


def _load_predicate_module():
    path = pathlib.Path(__file__).with_name("mla_head_padding.py")
    spec = importlib.util.spec_from_file_location(
        "_fa_cute_mla_head_padding_usability", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    # Make the sglang package unimportable so the sibling-module path is used.
    sys.modules["sglang"] = None
    module = _load_predicate_module()
    f = module.mla_2cta_pads_dense_qheads
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
