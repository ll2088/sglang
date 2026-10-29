"""Lightweight pytest bootstrap for the gRPC performance unit tests.

These tests only depend on the standard library and pytest.  When the full
``sglang`` package is not installed (e.g. CI runs the tests before
``pip install -e python``), install minimal stub modules so the target source
files can still be imported and exercised.

The stubs are intentionally tiny: they only provide the attributes the target
modules touch at import time.  If the real package is importable, nothing is
patched and the tests run against the real code.
"""

from __future__ import annotations

import importlib
import sys
import types


def _module_available(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


def _install_stub(name: str, attrs: dict | None = None) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in (attrs or {}).items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


def _bootstrap() -> None:
    if _module_available("sglang.srt.utils.protobuf_impl"):
        return

    # Parent packages: mark them as packages so submodule imports work.
    for pkg in ("sglang", "sglang.srt", "sglang.srt.utils"):
        if pkg not in sys.modules:
            stub = _install_stub(pkg)
            stub.__path__ = []  # type: ignore[attr-defined]

    # sglang.srt.utils.protobuf_impl
    def warn_if_python_protobuf() -> None:
        try:
            from google.protobuf.internal import api_implementation

            implementation = api_implementation.Type()
        except (ImportError, AttributeError):
            return
        if implementation == "python":
            import logging

            logging.getLogger(__name__).warning(
                "SGLang gRPC is using Python protobuf, which can reduce "
                "throughput. Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=upb "
                "before starting Python and install a supported protobuf "
                "wheel with the upb extension. Changing the environment "
                "after importing protobuf does not change the active "
                "implementation."
            )

    _install_stub(
        "sglang.srt.utils.protobuf_impl",
        {"warn_if_python_protobuf": warn_if_python_protobuf},
    )

    # sglang.srt.utils.streaming
    def token_history_len(history) -> int:
        return 0 if history is None else len(history)

    def token_history_view(history, start: int = 0):
        if history is None:
            return ()
        return history[start:]

    _install_stub(
        "sglang.srt.utils.streaming",
        {
            "token_history_len": token_history_len,
            "token_history_view": token_history_view,
        },
    )

    # sglang.srt.utils.async_queue
    import asyncio

    class EventLoopShutdownError(RuntimeError):
        pass

    def notify_one(loop: asyncio.AbstractEventLoop, callback) -> None:
        if not loop.is_running():
            raise EventLoopShutdownError()
        try:
            loop.call_soon_threadsafe(callback)
        except RuntimeError as error:
            if loop.is_closed():
                raise EventLoopShutdownError() from error
            raise

    def notify_many(loop: asyncio.AbstractEventLoop, callbacks) -> None:
        if not loop.is_running():
            raise EventLoopShutdownError()
        try:
            loop.call_soon_threadsafe(lambda: [cb() for cb in callbacks])
        except RuntimeError as error:
            if loop.is_closed():
                raise EventLoopShutdownError() from error
            raise

    _install_stub(
        "sglang.srt.utils.async_queue",
        {
            "EventLoopShutdownError": EventLoopShutdownError,
            "notify_one": notify_one,
            "notify_many": notify_many,
        },
    )


_bootstrap()
