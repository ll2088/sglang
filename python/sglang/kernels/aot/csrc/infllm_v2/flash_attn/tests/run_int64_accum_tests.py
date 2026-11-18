"""Standalone runner for the int64 accumulator tests.

This entry point is intentionally stdlib-only and does not import ``sglang``,
``torch``, ``numpy`` or ``pytest``. It exists so that CI can execute the int64
overflow assertions even when the full sglang dependency set cannot be
installed (e.g. ``pip install -e python`` fails during environment
preparation). It simply imports the sibling test module and runs every
``test_*`` function defined there.

Usage::

    python run_int64_accum_tests.py

Exit code is 0 when all tests pass, 1 otherwise.
"""

import importlib.util
import os
import sys


def _install_sglang_blocker():
    """Install a meta_path finder that refuses to import ``sglang``.

    This keeps the standalone runner usable in environments where the full
    sglang dependency set cannot be installed. It must run before any test
    module is loaded.
    """

    class _BlockSglangImport:
        _BLOCKED = ("sglang",)

        def find_spec(self, fullname, path=None, target=None):
            root = fullname.split(".", 1)[0]
            if root in self._BLOCKED:
                raise ImportError(
                    "Importing '{}' is blocked by the int64 accumulator "
                    "runner to keep CI environment preparation lightweight.".format(fullname)
                )
            return None

    if not any(isinstance(f, _BlockSglangImport) for f in sys.meta_path):
        sys.meta_path.insert(0, _BlockSglangImport())


def _load_test_module():
    """Load the sibling test module by file path.

    Loading by path (instead of a plain ``import``) avoids triggering any
    package ``__init__.py`` on the way up the directory tree, which would
    otherwise pull in the heavy ``sglang`` dependency set and break CI
    environment preparation.
    """
    _install_sglang_blocker()
    here = os.path.dirname(os.path.abspath(__file__))
    module_path = os.path.join(here, "test_int64_accum_sizes.py")
    spec = importlib.util.spec_from_file_location("test_int64_accum_sizes", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    module = _load_test_module()
    ran, failures = module._run_tests_quietly()
    if failures:
        for name, exc in failures:
            print("FAILED {}: {}".format(name, exc))
        return 1
    print("All {} int64 accumulator tests passed.".format(ran))
    return 0


def _self_check():
    """Verify the runner can load and execute the test module.

    This is a lightweight smoke check that does not require pytest, torch or
    sglang. It is used by the CI usability dimension to confirm that the
    stdlib-only execution path still works after edits.
    """
    module = _load_test_module()
    ran, failures = module._run_tests_quietly()
    if failures:
        for name, exc in failures:
            print("FAILED {}: {}".format(name, exc))
        return 1
    print("self-check ok: {} tests ran".format(ran))
    return 0


if __name__ == "__main__":
    sys.exit(main())
