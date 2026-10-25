"""Guards how LazyImport reports a broken lazily imported module.

An ``AttributeError`` raised while the target module is being imported is a
bug inside that module. ``hasattr()``, ``getattr(..., default)`` and
``from module import name`` all treat an ``AttributeError`` as "no such
name", so it must leave ``LazyImport`` as an ``ImportError`` chained to the
original error.

The fault is injected with a ``sys.meta_path`` finder that raises when the
target module is imported, so it covers both ``importlib.import_module`` and
``from ... import`` inside ``LazyImport``.
"""

import importlib
import importlib.abc
import sys

import pytest

from sglang.utils import LazyImport

_MARKER = "injected-by-test_lazy_import"


class _RaiseOnImport(importlib.abc.MetaPathFinder):
    def __init__(self, fullname: str):
        self.fullname = fullname
        self.attempts = 0

    def find_spec(self, name, path=None, target=None):
        if name != self.fullname:
            return None
        self.attempts += 1
        raise AttributeError(f"{_MARKER}: importing {name}")


@pytest.fixture
def broken_module(monkeypatch):
    """Make importing ``target`` raise AttributeError until the test ends."""

    def install(target: str):
        finder = _RaiseOnImport(target)
        monkeypatch.setattr(sys, "meta_path", [finder, *sys.meta_path])
        monkeypatch.delitem(sys.modules, target, raising=False)
        return finder

    return install


@pytest.mark.parametrize(
    "target,attr",
    [
        ("sglang._lazy_import_test_target", "Thing"),
        ("sglang._lazy_import_test_target", "submodule"),
    ],
    ids=["class-attr", "submodule-attr"],
)
def test_lazy_import_wraps_attribute_error(broken_module, target, attr):
    finder = broken_module(target)
    lazy = LazyImport(target, attr)

    with pytest.raises(ImportError) as excinfo:
        lazy()

    assert isinstance(excinfo.value.__cause__, AttributeError)
    assert _MARKER in str(excinfo.value.__cause__)
    assert _MARKER in str(excinfo.value)
    assert "AttributeError" in str(excinfo.value)
    assert finder.attempts == 1


def test_lazy_import_hasattr_does_not_swallow(broken_module):
    target = "sglang._lazy_import_test_target"
    broken_module(target)
    lazy = LazyImport(target, "Thing")

    with pytest.raises(ImportError) as excinfo:
        hasattr(lazy, "anything")

    assert isinstance(excinfo.value.__cause__, AttributeError)
    assert _MARKER in str(excinfo.value)


def test_lazy_import_getattr_default_does_not_swallow(broken_module):
    target = "sglang._lazy_import_test_target"
    broken_module(target)
    lazy = LazyImport(target, "Thing")

    with pytest.raises(ImportError) as excinfo:
        getattr(lazy, "anything", None)

    assert isinstance(excinfo.value.__cause__, AttributeError)
    assert _MARKER in str(excinfo.value)


def test_lazy_import_from_import_preserves_cause(broken_module):
    target = "sglang._lazy_import_test_target"
    finder = broken_module(target)
    lazy = LazyImport(target, "Thing")

    with pytest.raises(ImportError) as excinfo:
        from_import = getattr(lazy, "__getattr__", None)
        if from_import is not None:
            from_import("Thing")
        else:
            lazy()

    assert isinstance(excinfo.value.__cause__, AttributeError)
    assert _MARKER in str(excinfo.value)
    assert finder.attempts == 1


def test_lazy_import_missing_module_still_raises_import_error():
    lazy = LazyImport("sglang._definitely_missing_lazy_module", "Thing")

    with pytest.raises(ImportError) as excinfo:
        lazy()

    assert not isinstance(excinfo.value.__cause__, AttributeError)
    assert "_definitely_missing_lazy_module" in str(excinfo.value)
