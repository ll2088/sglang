"""Lightweight test bootstrap for the protobuf/streaming/queue unit tests.

These tests only need the standard library plus pytest.  When the full
``sglang`` dependency tree is not installed (e.g. in a minimal CI job), we
install stub modules for the heavy imports so the target source files can be
imported and exercised in isolation.  The stubs are intentionally tiny and
only cover the attributes the tests touch.
"""

from __future__ import annotations

import sys
import types


def _install_stub(name: str, **attrs) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module


def _bootstrap() -> None:
    # ``torch`` is imported by several entrypoint modules at import time.
    _install_stub("torch")
    # ``msgspec`` is used by managers/utils.py.
    _install_stub("msgspec")
    # ``psutil`` is used by rust_server/server.py.
    _install_stub("psutil")
    # ``sglang.srt.utils.protobuf_impl`` is the real target; do not stub it.
    # ``sglang.srt.utils.streaming`` is the real target; do not stub it.
    # ``sglang.srt.utils.async_queue`` is the real target; do not stub it.


_bootstrap()
