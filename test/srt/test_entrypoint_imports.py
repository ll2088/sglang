import importlib

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "sglang.srt.entrypoints.openai.utils",
        "sglang.srt.managers.utils",
        "sglang.srt.rust_server.server",
    ],
)
def test_entrypoint_modules_import(module):
    importlib.import_module(module)
