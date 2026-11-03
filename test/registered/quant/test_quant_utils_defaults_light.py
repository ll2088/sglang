# SPDX-License-Identifier: Apache-2.0
"""Dependency-free regression tests for mutable-default and module-level
state fixes.

These tests only import the pure-python helpers so they can run in a minimal
CI environment (pytest + stdlib) without torch or compressed_tensors.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_module(rel_path: str, name: str):
    path = REPO_ROOT / rel_path
    if not path.is_file():
        pytest.skip(f"missing source file: {rel_path}")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return tree


def test_no_mutable_defaults_in_quant_utils():
    tree = _load_module(
        "python/sglang/srt/layers/quantization/utils.py", "quant_utils"
    )
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for default in list(node.args.defaults) + [
            d for d in node.args.kw_defaults if d is not None
        ]:
            if isinstance(default, (ast.Dict, ast.List, ast.Set)):
                offenders.append(f"{node.name}@{node.lineno}")
    assert offenders == [], f"mutable defaults found: {offenders}"


def test_fallback_fused_shards_is_not_mutated_by_callers():
    from sglang.srt.layers.quantization.utils import (
        _FALLBACK_FUSED_SHARDS,
        is_layer_skipped,
    )

    before = {k: list(v) for k, v in _FALLBACK_FUSED_SHARDS.items()}
    is_layer_skipped("model.layers.0.mlp.gate_up_proj", ["mlp.gate"])
    is_layer_skipped("model.layers.0.mlp.gate_up_proj", ["mlp.gate"])
    assert {k: list(v) for k, v in _FALLBACK_FUSED_SHARDS.items()} == before


def test_is_layer_skipped_accepts_none_fused_mapping():
    from sglang.srt.layers.quantization.utils import is_layer_skipped

    assert is_layer_skipped("model.layers.0.self_attn.q_proj", ["q_proj"], None)
    assert not is_layer_skipped("model.layers.0.self_attn.q_proj", [], None)
