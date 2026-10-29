"""Entrypoint import smoke tests (standard library only)."""

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]

ENTRYPOINTS = [
    "python/sglang/srt/entrypoints/openai/utils.py",
    "python/sglang/srt/managers/utils.py",
    "python/sglang/srt/rust_server/server.py",
]


@pytest.mark.parametrize("relative", ENTRYPOINTS)
 def test_entrypoint_parses(relative):
    path = ROOT / relative
    assert path.is_file(), relative
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
