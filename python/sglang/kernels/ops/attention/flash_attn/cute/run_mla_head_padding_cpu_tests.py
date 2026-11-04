#!/usr/bin/env python3
"""CPU-only entry point for the dense 2CTA MLA Q-head padding rule.

This script deliberately imports nothing beyond the standard library and
pytest. It loads ``mla_head_padding.py`` and
``test_mla_head_padding_standalone.py`` by file path so it can run on a
CPU-only runner without CUDA/CuTe, torch, quack or the sglang package, and
without installing the full sglang dependency tree.

Usage::

    python run_mla_head_padding_cpu_tests.py
    python run_mla_head_padding_cpu_tests.py --pytest-args "-q"
"""

import argparse
import importlib.util
import pathlib
import sys


_HERE = pathlib.Path(__file__).resolve().parent
_PREDICATE_PATH = _HERE / "mla_head_padding.py"
_STANDALONE_TEST_PATH = _HERE / "test_mla_head_padding_standalone.py"


def _load_module_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check_predicate():
    module = _load_module_by_path("_fa_cute_mla_head_padding_cpu", _PREDICATE_PATH)
    pads = module.mla_2cta_pads_dense_qheads
    accepts = module.mla_2cta_accepts_dense_qheads

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
    assert pads(127, True) is True
    assert pads(128, True) is False
    assert pads(96, False) is False
    assert pads(0, True) is False
    assert pads(-1, True) is False

    # Acceptance rule: dividing heads as-is, or MQA 65-127 by padding.
    assert accepts(128, 1) is True
    assert accepts(64, 1) is True
    assert accepts(96, 1) is True
    assert accepts(65, 1) is True
    assert accepts(127, 1) is True
    assert accepts(24, 1) is False
    assert accepts(96, 2) is False
    assert accepts(0, 1) is False
    assert accepts(-1, 1) is False


def _run_standalone_pytest(extra_args):
    try:
        import pytest
    except ImportError as exc:  # pragma: no cover - environment issue
        raise SystemExit(
            "pytest is required to run the standalone tests: " f"{exc}"
        )
    args = [str(_STANDALONE_TEST_PATH), *extra_args]
    return pytest.main(args)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pytest-args",
        default="-q",
        help="extra arguments passed to pytest (default: -q)",
    )
    parser.add_argument(
        "--skip-pytest",
        action="store_true",
        help="only run the inline predicate assertions",
    )
    args = parser.parse_args(argv)

    _check_predicate()
    print("mla_head_padding predicate assertions: ok")

    if args.skip_pytest:
        return 0

    extra = args.pytest_args.split() if args.pytest_args else []
    return int(_run_standalone_pytest(extra))


if __name__ == "__main__":
    sys.exit(main())
