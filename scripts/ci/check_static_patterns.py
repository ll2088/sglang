#!/usr/bin/env python3
"""Static checks for the patterns fixed by the upstream port.

This script is intentionally dependency-free (stdlib only) so it can run in
CI before any package installation. It scans the files touched by the port
for:

* mutable default arguments (dict/list/set literals in signatures),
* in-place mutation of module-level dicts,
* ``hasattr`` checks on dataclass fields.

Exit code is non-zero when a violation is found.
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


def _check_mutable_defaults(tree: ast.AST, path: Path) -> list[str]:
    problems: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        args = node.args
        all_defaults = list(args.defaults) + [d for d in args.kw_defaults if d is not None]
        for default in all_defaults:
            if _is_mutable_default(default):
                problems.append(
                    f"{path}:{default.lineno}: mutable default in {node.name}()"
                )
    return problems


def _check_hasattr_on_dataclass(tree: ast.AST, path: Path) -> list[str]:
    problems: list[str] = []
    dataclass_fields: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            is_dataclass = any(
                (isinstance(d, ast.Name) and d.id == "dataclass")
                or (isinstance(d, ast.Attribute) and d.attr == "dataclass")
                for d in node.decorator_list
            )
            if not is_dataclass:
                continue
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    dataclass_fields.add(stmt.target.id)
    if not dataclass_fields:
        return problems
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Name) and func.id == "hasattr"):
            continue
        if len(node.args) < 2:
            continue
        attr = node.args[1]
        if isinstance(attr, ast.Constant) and attr.value in dataclass_fields:
            problems.append(
                f"{path}:{node.lineno}: hasattr() on dataclass field {attr.value!r}"
            )
    return problems


def _check_module_dict_mutation(tree: ast.AST, path: Path) -> list[str]:
    problems: list[str] = []
    module_dicts: set[str] = set()
    for node in tree.body if isinstance(tree, ast.Module) else []:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and isinstance(node.value, ast.Dict):
                    module_dicts.add(target.id)
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
                f"{path}:{node.lineno}: in-place mutation of module dict {value.id!r}"
            )
    return problems


def main() -> int:
    all_problems: list[str] = []
    for rel in TARGET_FILES:
        path = REPO_ROOT / rel
        if not path.exists():
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            all_problems.append(f"{path}: syntax error: {exc}")
            continue
        all_problems.extend(_check_mutable_defaults(tree, path))
        all_problems.extend(_check_hasattr_on_dataclass(tree, path))
        all_problems.extend(_check_module_dict_mutation(tree, path))

    if all_problems:
        for problem in all_problems:
            print(problem, file=sys.stderr)
        return 1
    print("static pattern checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
