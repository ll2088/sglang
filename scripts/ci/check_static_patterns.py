#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Static checks for patterns fixed by the upstream SonarQube PR.

This script is intentionally dependency-free so it can run before the full
test environment is installed. It scans the files touched by the port for:

* mutable default arguments (dict/list/set literals in signatures)
* ``hasattr`` checks on dataclass fields
* in-place mutation of module-level mapping constants

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


def check_mutable_defaults(path: Path, tree: ast.AST) -> list[str]:
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


def check_hasattr_on_dataclass(path: Path, tree: ast.AST) -> list[str]:
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
                f"{path}:{node.lineno}: hasattr on dataclass field {attr.value!r}"
            )
    return problems


def check_module_level_mutation(path: Path, tree: ast.AST) -> list[str]:
    problems: list[str] = []
    module_constants: set[str] = set()
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    module_constants.add(target.id)
    if not module_constants:
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
        if isinstance(value, ast.Name) and value.id in module_constants:
            problems.append(
                f"{path}:{node.lineno}: in-place mutation of module constant {value.id}"
            )
    return problems


def main() -> int:
    all_problems: list[str] = []
    for rel in TARGET_FILES:
        path = REPO_ROOT / rel
        if not path.exists():
            print(f"skip (missing): {rel}")
            continue
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        all_problems.extend(check_mutable_defaults(path, tree))
        all_problems.extend(check_hasattr_on_dataclass(path, tree))
        all_problems.extend(check_module_level_mutation(path, tree))

    if all_problems:
        print("Static pattern violations found:")
        for problem in all_problems:
            print(f"  {problem}")
        return 1

    print("No mutable defaults, dataclass hasattr misuse, or module-level mutation found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
