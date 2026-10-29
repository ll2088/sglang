"""Minimal test bootstrap for the gRPC/SMG performance unit tests.

These tests only need the standard library, ``pytest`` and (optionally) the
``protobuf`` runtime. The full ``sglang`` dependency tree is not required, so
this conftest installs lightweight stub modules for the few ``sglang``
packages the target sources import at module scope. Real modules win: a stub
is only registered when the import would otherwise fail.
"""

from __future__ import annotations

import importlib
import sys
import types


def _ensure_package(name: str) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = []  # mark as a package
        sys.modules[name] = module
    return module


def _try_import(name: str) -> bool:
    try:
        importlib.import_module(name)
    except Exception:
        return False
    return True


def _install_stub(name: str, **attrs) -> None:
    if _try_import(name):
        return
    parent, _, leaf = name.rpartition(".")
    if parent:
        _ensure_package(parent)
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    if parent:
        setattr(sys.modules[parent], leaf, module)


# ``sglang.srt.utils.protobuf_impl`` only needs the standard library, but the
# package ``__init__`` files pull in heavy runtime dependencies. Register the
# package chain as empty packages so the real submodule can be imported.
for _pkg in (
    "sglang",
    "sglang.srt",
    "sglang.srt.utils",
    "sglang.srt.entrypoints",
    "sglang.srt.entrypoints.openai",
    "sglang.srt.managers",
):
    _ensure_package(_pkg)

# ``sglang.srt.entrypoints.openai.utils`` imports torch and the OpenAI protocol
# models at module scope; the import test only checks that the module loads.
_install_stub("torch", Tensor=object)
_install_stub(
    "sglang.srt.entrypoints.openai.protocol",
    CachedTokensDetails=object,
    ChatCompletionRequest=object,
    CompletionRequest=object,
    LogProbs=object,
    SpecTokensDetails=object,
    StreamOptions=object,
)

# ``sglang.srt.managers.utils`` imports msgspec and torch at module scope.
_install_stub("msgspec", Struct=object, msgpack=types.SimpleNamespace())
_install_stub("sglang.srt.constants", HEALTH_CHECK_RID_PREFIX="HEALTH_CHECK")
_install_stub("sglang.srt.managers.io_struct", msgpack_decode=lambda data: data)
_install_stub("sglang.srt.managers.schedule_batch", Req=object)
