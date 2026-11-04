#!/usr/bin/env python3
# Copyright (c) 2025, Tri Dao.
"""Stdlib-only runner for the dense 2CTA MLA Q-head padding tests.

This entry point deliberately avoids importing the sglang package, torch,
cutlass or quack. It loads ``mla_head_padding.py`` and
``test_mla_head_padding_standalone.py`` by file path so a minimal CI job can
validate the padding rule without installing the full sglang dependency tree.

Usage::

    python run_mla_head_padding_cpu_tests.py

Exit code is 0 when every check passes, 1 otherwise.
"""

import importlib.util
import pathlib
import sys
import traceback


_HERE = pathlib.Path(__file__).resolve().parent


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_standalone_tests():
    """Run the pytest-style standalone module without requiring pytest.

    The standalone module exposes plain functions decorated with
    ``pytest.mark.parametrize``. When pytest is unavailable we cannot expand
    those parametrizations, so we fall back to the explicit boundary checks
    below. When pytest is available we delegate to it.
    """
    try:
        import pytest  # noqa: F401
    except ImportError:
        return None
    return pytest.main(
        [
            "-q",
            "-p",
            "no:cacheprovider",
            str(_HERE / "test_mla_head_padding_standalone.py"),
        ]
    )


def _check_boundaries(predicate, accepts):
    cases = [
        (64, 1, False),
        (65, 1, True),
        (96, 1, True),
        (127, 1, True),
        (128, 1, False),
        (24, 1, False),
        (0, 1, False),
        (-1, 1, False),
        (129, 1, False),
        (96, 2, False),
        (65, 2, False),
        (96, 0, False),
        (64, True, False),
        (65, True, True),
        (96, True, True),
        (127, True, True),
        (128, True, False),
        (0, True, False),
        (-1, True, False),
        (96, False, False),
    ]
    failures = []
    for nheads, heads_kv, expected in cases:
        got = predicate(nheads, heads_kv)
        if got is not expected:
            failures.append(
                f"mla_2cta_pads_dense_qheads({nheads}, {heads_kv}) = {got!r}, "
                f"expected {expected!r}"
            )

    accept_cases = [
        (128, 1, True),
        (64, 1, True),
        (96, 1, True),
        (65, 1, True),
        (127, 1, True),
        (24, 1, False),
        (96, 2, False),
        (0, 1, False),
        (-1, 1, False),
    ]
    for nheads, heads_kv, expected in accept_cases:
        got = accepts(nheads, heads_kv)
        if got is not expected:
            failures.append(
                f"mla_2cta_accepts_dense_qheads({nheads}, {heads_kv}) = {got!r}, "
                f"expected {expected!r}"
            )
    return failures


def main():
    module = _load_module(
        "_fa_cute_mla_head_padding_cpu_runner", _HERE / "mla_head_padding.py"
    )
    predicate = module.mla_2cta_pads_dense_qheads
    accepts = module.mla_2cta_accepts_dense_qheads

    failures = _check_boundaries(predicate, accepts)
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1

    rc = _run_standalone_tests()
    if rc is None:
        print("pytest not available; boundary checks passed")
        return 0
    return int(rc)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # pragma: no cover - surface the traceback in CI logs
        traceback.print_exc()
        sys.exit(1)
