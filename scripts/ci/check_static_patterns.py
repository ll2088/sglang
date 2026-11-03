#!/usr/bin/env python3
"""Static checks for patterns fixed by the upstream SonarQube PR.

This script is intentionally dependency-free so it can run before the test
dependencies are installed. It scans the files touched by the port for:

* mutable default arguments (dict/list/set literals in signatures)
* in-place mutation of module-level mapping constants
* ``hasattr`` checks on dataclass optional fields
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGETS = [
    ROOT / "python/sglang/srt/layers/quantization/utils.py",
    ROOT / "python/sglang/srt/layers/quantization/compressed_tensors/utils.py",
    ROOT / "python/sglang/srt/layers/quantization/quark/utils.py",
    ROOT / "python/sglang/srt/models/inkling_common/quantization/config.py",
    ROOT / "python/sglang/srt/utils/hf_transformers/config.py",
    ROOT / "python/sglang/srt/model_loader/utils.py",
]

MUTABLE_LITERALS = (ast.Dict, ast.List, ast.Set)


def _is_mutable_default(node: ast.AST) -> bool:
    return isinstance(node, MUTABLE_LITERALS)


def check_mutable_defaults(path: Path, tree: ast.AST) -> list[str]:
    problems: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        args = node.args
        defaults = list(args.defaults)
        for default in defaults:
            if _is_mutable_default(default):
                problems.append(
                    f"{path}:{node.lineno}: mutable default in {node.name}()"
                )
        for default in args.kw_defaults:
            if default is not None and _is_mutable_default(default):
                problems.append(
                    f"{path}:{node.lineno}: mutable keyword default in {node.name}()"
                )
    return problems


def check_module_level_mutation(path: Path, tree: ast.AST) -> list[str]:
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
        if func.attr not in {"update", "append", "extend", "insert", "pop", "clear"}:
            continue
        value = func.value
        if isinstance(value, ast.Name) and value.id in module_constants:
            problems.append(
                f"{path}:{node.lineno}: in-place mutation of module constant {value.id}"
            )
    return problems


def check_hasattr_on_dataclass_fields(path: Path, tree: ast.AST) -> list[str]:
    problems: list[str] = []
    dataclass_fields: set[str] = set()
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
        if len(node.args) != 2:
            continue
        attr = node.args[1]
        if isinstance(attr, ast.Constant) and attr.value in dataclass_fields:
            problems.append(
                f"{path}:{node.lineno}: hasattr on dataclass field {attr.value!r}"
            )
    return problems


def main() -> int:
    problems: list[str] = []
    for path in TARGETS:
        if not path.exists():
            problems.append(f"missing target file: {path}")
            continue
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        problems.extend(check_mutable_defaults(path, tree))
        problems.extend(check_module_level_mutation(path, tree))
        problems.extend(check_hasattr_on_dataclass_fields(path, tree))
    if problems:
        print("static pattern check failed:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    print("static pattern check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
