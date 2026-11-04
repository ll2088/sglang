# Copyright (c) 2025, Tri Dao.
"""Stdlib-only entry point for the dense 2CTA MLA Q-head padding tests.

This script exists so CI can validate the pure-Python padding rule without
installing the full sglang dependency tree (torch, quack, cutlass, ...). It
loads ``mla_head_padding.py`` and ``test_mla_head_padding_standalone.py`` by
file path and runs the tests with pytest if available, otherwise with a tiny
built-in runner that understands the ``@pytest.mark.parametrize`` decorator
used by the standalone module.

Usage::

    python run_mla_head_padding_cpu_tests.py

Exit code 0 means every case passed.
"""

import importlib.util
import pathlib
import sys
import traceback


_HERE = pathlib.Path(__file__).resolve().parent


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_standalone():
    return _load_module(
        "_fa_cute_mla_head_padding_standalone",
        _HERE / "test_mla_head_padding_standalone.py",
    )


def _run_with_pytest():
    import pytest

    return pytest.main(
        [
            "-q",
            "-p",
            "no:cacheprovider",
            str(_HERE / "test_mla_head_padding_standalone.py"),
        ]
    )


def _iter_parametrized_cases(func):
    """Yield (id, kwargs) for a function decorated with pytest.mark.parametrize.

    Only the simple form used by the standalone module is supported: a single
    ``parametrize`` mark whose argnames are a tuple/list of strings and whose
    argvalues are tuples. Anything else is reported as unsupported so the
    caller can fall back to pytest.
    """
    marks = getattr(func, "pytestmark", None)
    if not marks:
        return None
    for mark in marks:
        if getattr(mark, "name", None) != "parametrize":
            continue
        args = getattr(mark, "args", ())
        if len(args) != 2:
            return None
        argnames, argvalues = args
        if isinstance(argnames, str):
            argnames = [a.strip() for a in argnames.split(",") if a.strip()]
        else:
            argnames = list(argnames)
        cases = []
        for values in argvalues:
            if not isinstance(values, (tuple, list)):
                values = (values,)
            if len(values) != len(argnames):
                return None
            cases.append(dict(zip(argnames, values)))
        return cases
    return None


def _run_without_pytest():
    module = _load_standalone()
    failures = []
    total = 0
    for name in sorted(dir(module)):
        if not name.startswith("test_"):
            continue
        func = getattr(module, name)
        if not callable(func):
            continue
        cases = _iter_parametrized_cases(func)
        if cases is None:
            # No parametrize mark: call with no arguments.
            cases = [{}]
        for kwargs in cases:
            total += 1
            case_id = f"{name}{tuple(kwargs.values()) if kwargs else ''}"
            try:
                func(**kwargs)
            except Exception:  # noqa: BLE001 - report every failure
                failures.append((case_id, traceback.format_exc()))
    if failures:
        for case_id, tb in failures:
            sys.stderr.write(f"FAILED {case_id}\n{tb}\n")
        sys.stderr.write(f"{len(failures)} of {total} cases failed\n")
        return 1
    sys.stdout.write(f"{total} cases passed\n")
    return 0


def main():
    try:
        import pytest  # noqa: F401
    except ImportError:
        return _run_without_pytest()
    return _run_with_pytest()


if __name__ == "__main__":
    sys.exit(main())
