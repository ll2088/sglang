"""Import smoke tests for entrypoint modules touched by the perf port.

Only the dependency-free modules are imported here; the heavier entrypoint
modules are checked for syntax/compile validity so the lightweight CI job
can still catch accidental breakage without installing torch.
"""

from __future__ import annotations

import py_compile
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _REPO_ROOT / "python"

_LIGHT_MODULES = [
    "sglang.srt.utils.protobuf_impl",
    "sglang.srt.utils.streaming",
    "sglang.srt.utils.async_queue",
]

_COMPILE_ONLY = [
    "sglang/srt/entrypoints/openai/utils.py",
    "sglang/srt/managers/utils.py",
    "sglang/srt/rust_server/server.py",
]


@pytest.mark.parametrize("module", _LIGHT_MODULES)
def test_light_module_imports(module: str) -> None:
    __import__(module)


@pytest.mark.parametrize("relative", _COMPILE_ONLY)
def test_entrypoint_module_compiles(relative: str) -> None:
    path = _SRC_ROOT / relative
    if not path.exists():
        pytest.skip(f"{relative} not present in this checkout")
    py_compile.compile(str(path), doraise=True)
