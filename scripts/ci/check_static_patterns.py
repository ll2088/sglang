#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Static checks for the patterns fixed by the upstream port.

Scans the touched production files for:
  * mutable default arguments (dict/list/set literals in signatures)
  * ``hasattr`` used on dataclass optional fields
  * in-place mutation of module-level mapping constants

Exits non-zero when a violation is found so CI can gate on it.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

TARGET_FILES = [
    "python/sglang/srt/layers/quantization/utils.py",
    "python/sglang/srt/layers/quantization/compressed_tensors/utils.py",
    "python/sglang/srt/layers/quantization/quark/utils.py",
    "python/sglang/srt/models/inkling_common/quantization/config.py",
    "python/sglang/srt/utils/hf_transformers/config.py",
    "python/sglang/srt/model_loader/utils.py",
]

MUTABLE_LITERALS = (ast.Dict, ast.List, ast.Set)


def _is_mutable_default(node: ast.AST) -> bool:
    return isinstance(node, MUTABLE_LITERALS)


def check_mutable_defaults(tree: ast.AST, path: str) -> list[str]:
    problems: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        args = node.args
        all_defaults = list(args.defaults) + [d for d in args.kw_defaults if d]
        for default in all_defaults:
            if _is_mutable_default(default):
                problems.append(
                    f"{path}:{node.lineno}: mutable default in {node.name}()"
                )
    return problems


def check_hasattr_on_dataclass_fields(tree: ast.AST, path: str) -> list[str]:
    problems: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Name) and func.id == "hasattr"):
            continue
        if len(node.args) != 2:
            continue
        attr = node.args[1]
        if isinstance(attr, ast.Constant) and isinstance(attr.value, str):
            name = attr.value
            if name in {"vocab_embedding", "lm_head"}:
                problems.append(
                    f"{path}:{node.lineno}: hasattr on dataclass field {name!r}"
                )
    return problems


def check_module_level_mutation(tree: ast.AST, path: str) -> list[str]:
    problems: list[str] = []
    module_constants = {
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id.isupper()
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        if func.attr not in {"update", "append", "extend", "setdefault", "pop"}:
            continue
        value = func.value
        if isinstance(value, ast.Name) and value.id in module_constants:
            problems.append(
                f"{path}:{node.lineno}: in-place mutation of module constant "
                f"{value.id}"
            )
    return problems


def main() -> int:
    problems: list[str] = []
    for rel in TARGET_FILES:
        path = REPO_ROOT / rel
        if not path.exists():
            continue
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=rel)
        problems.extend(check_mutable_defaults(tree, rel))
        problems.extend(check_hasattr_on_dataclass_fields(tree, rel))
        problems.extend(check_module_level_mutation(tree, rel))

    if problems:
        print("Static pattern violations found:")
        for problem in problems:
            print(f"  {problem}")
        return 1

    print("No mutable-default / hasattr / module-mutation violations found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
