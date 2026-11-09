"""Make pure-helper OpenAI entrypoint tests collectable without the full
sglang runtime dependency chain installed.

CI installs the package with its full dependency set, but that step can fail
before pytest ever runs. These tests only exercise protocol models and pure
helpers, so we pre-register lightweight package stubs for the heavy runtime
modules. Real modules are still preferred when importable: the stub is only
installed for names that are not already in ``sys.modules`` and whose import
fails.
"""

from __future__ import annotations

import importlib
import sys
import types

_STUBBED = (
    "sglang.srt.managers.tokenizer_manager",
    "sglang.srt.managers.io_struct",
    "sglang.srt.parser.template_manager",
    "sglang.srt.parser.code_completion_parser",
    "sglang.srt.runtime_context",
    "sglang.srt.utils.weight_versions",
    "sglang.srt.entrypoints.openai.serving_base",
    "sglang.srt.entrypoints.openai.usage_processor",
    "sglang.srt.entrypoints.openai.utils",
    "sglang.utils",
)


def _ensure_package(name: str) -> None:
    if name in sys.modules:
        return
    mod = types.ModuleType(name)
    mod.__path__ = []
    sys.modules[name] = mod


def _stub(name: str) -> None:
    if name in sys.modules:
        return
    try:
        importlib.import_module(name)
        return
    except Exception:
        pass
    parent, _, _ = name.rpartition(".")
    if parent:
        _ensure_package(parent)
    mod = types.ModuleType(name)
    mod.__getattr__ = lambda attr: _StubAttr(name, attr)
    sys.modules[name] = mod


class _StubAttr:
    """Placeholder for any attribute accessed on a stubbed module."""

    def __init__(self, module: str, attr: str):
        self._module = module
        self._attr = attr

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            f"{self._module}.{self._attr} is stubbed in unit tests"
        )

    def __getattr__(self, item):
        return _StubAttr(self._module, f"{self._attr}.{item}")


for _name in ("sglang", "sglang.srt", "sglang.srt.entrypoints",
              "sglang.srt.entrypoints.openai"):
    _ensure_package(_name)

for _name in _STUBBED:
    _stub(_name)
