"""Import smoke tests for entrypoints touched by the gRPC perf work."""

import importlib

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "sglang.srt.entrypoints.openai.utils",
        "sglang.srt.managers.utils",
        "sglang.srt.rust_server.server",
        "sglang.srt.utils.protobuf_impl",
        "sglang.srt.utils.streaming",
        "sglang.srt.utils.async_queue",
    ],
)
def test_module_imports(module: str) -> None:
    importlib.import_module(module)
