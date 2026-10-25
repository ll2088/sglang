"""Standalone verification of LazyImport error semantics (no pytest required).

An ``AttributeError`` raised while the target module is being imported is a
bug inside that module. ``hasattr()``, ``getattr(..., default)`` and
``from module import name`` all treat an ``AttributeError`` as "no such
name", so it must leave ``LazyImport`` as an ``ImportError`` chained to the
original error.

This script mirrors ``test_lazy_import.py`` but runs with the standard
library only, so it can be executed in environments where pytest is not
installed. It exits non-zero on the first failure.
"""

import importlib
import importlib.abc
import sys

from sglang.utils import LazyImport

_MARKER = "injected-by-verify_lazy_import_errors"


class _RaiseOnImport(importlib.abc.MetaPathFinder):
    def __init__(self, fullname: str):
        self.fullname = fullname
        self.attempts = 0

    def find_spec(self, name, path=None, target=None):
        if name != self.fullname:
            return None
        self.attempts += 1
        raise AttributeError(f"{_MARKER}: importing {name}")


class _BrokenModule:
    def __init__(self, target: str):
        self.target = target
        self.finder = _RaiseOnImport(target)

    def __enter__(self):
        self._saved_meta_path = sys.meta_path
        self._saved_module = sys.modules.pop(self.target, None)
        sys.meta_path = [self.finder, *sys.meta_path]
        return self.finder

    def __exit__(self, exc_type, exc, tb):
        sys.meta_path = self._saved_meta_path
        sys.modules.pop(self.target, None)
        if self._saved_module is not None:
            sys.modules[self.target] = self._saved_module
        return False


def _assert_chained_import_error(exc: BaseException, finder: _RaiseOnImport) -> None:
    assert isinstance(exc, ImportError), f"expected ImportError, got {type(exc)!r}"
    assert isinstance(
        exc.__cause__, AttributeError
    ), f"expected AttributeError cause, got {type(exc.__cause__)!r}"
    assert _MARKER in str(exc.__cause__), "cause message lost"
    assert _MARKER in str(exc), "original message not included in ImportError"
    assert "AttributeError" in str(exc), "original exception type not mentioned"
    assert finder.attempts == 1, f"expected single import attempt, got {finder.attempts}"


def _check_direct_call(target: str, attr: str) -> None:
    with _BrokenModule(target) as finder:
        lazy = LazyImport(target, attr)
        try:
            lazy()
        except ImportError as e:
            _assert_chained_import_error(e, finder)
        else:
            raise AssertionError("LazyImport did not raise")


def _check_hasattr(target: str, attr: str) -> None:
    with _BrokenModule(target) as finder:
        lazy = LazyImport(target, attr)
        try:
            hasattr(lazy, "anything")
        except ImportError as e:
            _assert_chained_import_error(e, finder)
        else:
            raise AssertionError("hasattr swallowed the failure")


def _check_getattr_default(target: str, attr: str) -> None:
    with _BrokenModule(target) as finder:
        lazy = LazyImport(target, attr)
        try:
            getattr(lazy, "anything", None)
        except ImportError as e:
            _assert_chained_import_error(e, finder)
        else:
            raise AssertionError("getattr(default) swallowed the failure")


def _check_missing_module() -> None:
    lazy = LazyImport("sglang._definitely_missing_lazy_module", "Thing")
    try:
        lazy()
    except ImportError as e:
        assert not isinstance(
            e.__cause__, AttributeError
        ), "missing module must not be reported as AttributeError"
        assert "_definitely_missing_lazy_module" in str(e)
    else:
        raise AssertionError("missing module did not raise ImportError")


def _check_public_lazy_exports() -> None:
    """Every LazyImport exported by sglang/__init__.py must wrap AttributeError."""
    import sglang

    checked = 0
    for name in sglang.__all__:
        value = getattr(sglang, name)
        if not isinstance(value, LazyImport):
            continue
        target = value.module_name
        attr = value.name
        with _BrokenModule(target) as finder:
            lazy = LazyImport(target, attr)
            try:
                lazy()
            except ImportError as e:
                _assert_chained_import_error(e, finder)
            else:
                raise AssertionError(f"LazyImport for {name!r} did not raise")
        checked += 1
    assert checked > 0, "no LazyImport exports found in sglang.__all__"


def main() -> int:
    target = "sglang._lazy_import_test_target"
    for attr in ("Thing", "submodule"):
        _check_direct_call(target, attr)
        _check_hasattr(target, attr)
        _check_getattr_default(target, attr)
    _check_missing_module()
    _check_public_lazy_exports()
    print("lazy import error semantics verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
