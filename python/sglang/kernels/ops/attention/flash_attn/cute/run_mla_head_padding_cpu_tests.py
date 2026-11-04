"""CPU-only runner for the dense 2CTA MLA Q-head padding predicate.

This script deliberately imports only the standard library and pytest. It
loads ``mla_head_padding.py`` and ``test_mla_head_padding_standalone.py`` by
file path so it can run on a minimal CI job without installing sglang, torch,
cutlass or quack.

Usage::

    python run_mla_head_padding_cpu_tests.py
    python run_mla_head_padding_cpu_tests.py -q --count=3

Any extra arguments are forwarded to pytest.
"""

import importlib.util
import pathlib
import sys


_HERE = pathlib.Path(__file__).resolve().parent


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check_predicate():
    """Fail fast with a clear message if the predicate file is missing/broken."""
    predicate_path = _HERE / "mla_head_padding.py"
    if not predicate_path.is_file():
        raise SystemExit(f"missing predicate module: {predicate_path}")
    module = _load_module("_fa_cute_mla_head_padding_runner", predicate_path)
    predicate = module.mla_2cta_pads_dense_qheads
    cases = [
        (64, True, False),
        (65, True, True),
        (127, True, True),
        (128, True, False),
        (96, False, False),
        (0, True, False),
        (-1, True, False),
    ]
    for nheads, mqa, expected in cases:
        got = predicate(nheads, mqa)
        if got is not expected:
            raise SystemExit(
                f"predicate({nheads}, {mqa}) = {got!r}, expected {expected!r}"
            )


def main(argv):
    _check_predicate()
    standalone = _HERE / "test_mla_head_padding_standalone.py"
    if not standalone.is_file():
        raise SystemExit(f"missing standalone test module: {standalone}")
    import pytest

    args = [str(standalone), *argv]
    return pytest.main(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
