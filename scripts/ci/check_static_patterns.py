#!/usr/bin/env python3
"""Static review for the patterns fixed by the upstream port.

Checks (over the files touched by this change set):
  1. mutable default arguments (dict/list/set literals in signatures)
  2. in-place mutation of module-level dicts
  3. `hasattr` used on dataclass optional fields

Exits non-zero when a pattern is found so CI can gate on it.
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
        all_defaults = list(args.defaults) + [d for d in args.kw_defaults if d is not None]
        for default in all_defaults:
            if _is_mutable_default(default):
                problems.append(
                    f"{path}:{node.lineno}: mutable default in '{node.name}'"
                )
    return problems


def check_module_dict_mutation(tree: ast.AST, path: str) -> list[str]:
    problems: list[str] = []
    module_dicts: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if isinstance(node.value, ast.Dict):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        module_dicts.add(target.id)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.value, ast.Dict) and isinstance(node.target, ast.Name):
                module_dicts.add(node.target.id)

    if not module_dicts:
        return problems

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        if func.attr not in {"update", "setdefault", "pop", "clear"}:
            continue
        value = func.value
        if isinstance(value, ast.Name) and value.id in module_dicts:
            problems.append(
                f"{path}:{node.lineno}: in-place mutation of module-level dict '{value.id}'"
            )
    return problems


def check_dataclass_hasattr(tree: ast.AST, path: str) -> list[str]:
    problems: list[str] = []
    dataclass_fields: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        is_dataclass = any(
            (isinstance(d, ast.Name) and d.id == "dataclass")
            or (isinstance(d, ast.Attribute) and d.attr == "dataclass")
            for d in node.decorator_list
        )
        if not is_dataclass:
            continue
        fields: set[str] = set()
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                fields.add(stmt.target.id)
        if fields:
            dataclass_fields[node.name] = fields

    if not dataclass_fields:
        return problems

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Name) and func.id == "hasattr"):
            continue
        if len(node.args) != 2:
            continue
        attr = node.args[1]
        if not (isinstance(attr, ast.Constant) and isinstance(attr.value, str)):
            continue
        for cls_name, fields in dataclass_fields.items():
            if attr.value in fields:
                problems.append(
                    f"{path}:{node.lineno}: hasattr on dataclass field "
                    f"'{cls_name}.{attr.value}' (use 'is not None')"
                )
    return problems


def main() -> int:
    problems: list[str] = []
    checked = 0
    for rel in TARGET_FILES:
        path = REPO_ROOT / rel
        if not path.is_file():
            continue
        checked += 1
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=rel)
        except SyntaxError as exc:
            problems.append(f"{rel}: syntax error: {exc}")
            continue
        problems.extend(check_mutable_defaults(tree, rel))
        problems.extend(check_module_dict_mutation(tree, rel))
        problems.extend(check_dataclass_hasattr(tree, rel))

    if checked == 0:
        print("ERROR: no target files found; check paths", file=sys.stderr)
        return 1

    if problems:
        print("Static pattern check failed:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1

    print(f"Static pattern check passed ({checked} files).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
