#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Static grep-style checks for the patterns fixed in this port.

This script is intentionally dependency-free so it can run in the minimal CI
environment. It scans the files touched by the port and fails if any of the
following patterns reappear:

* mutable default arguments (``=[]``, ``={}``, ``=set()``) in function or
  method signatures;
* ``hasattr`` checks on dataclass optional fields;
* in-place mutation of module-level mapping constants.
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

MUTABLE_DEFAULT_NODES = (ast.List, ast.Dict, ast.Set)


def _is_mutable_default(node: ast.AST) -> bool:
    if isinstance(node, MUTABLE_DEFAULT_NODES):
        return True
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        return node.func.id in {"list", "dict", "set"}
    return False


def check_mutable_defaults(path: Path, tree: ast.AST) -> list[str]:
    errors: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        args = node.args
        positional = list(args.posonlyargs) + list(args.args)
        defaults = list(args.defaults)
        if defaults:
            offset = len(positional) - len(defaults)
            for arg, default in zip(positional[offset:], defaults):
                if _is_mutable_default(default):
                    errors.append(
                        f"{path}:{node.lineno}: mutable default for "
                        f"argument '{arg.arg}' in {node.name}()"
                    )
        for arg, default in zip(args.kwonlyargs, args.kw_defaults):
            if default is not None and _is_mutable_default(default):
                errors.append(
                    f"{path}:{node.lineno}: mutable default for "
                    f"keyword argument '{arg.arg}' in {node.name}()"
                )
    return errors


def check_hasattr_on_dataclass_fields(path: Path, tree: ast.AST) -> list[str]:
    errors: list[str] = []
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
        return errors
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
            errors.append(
                f"{path}:{node.lineno}: hasattr() on dataclass field "
                f"'{attr.value}'; use 'is not None' instead"
            )
    return errors


def check_module_level_mapping_mutation(path: Path, tree: ast.AST) -> list[str]:
    errors: list[str] = []
    module_mappings: set[str] = set()
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name) and isinstance(stmt.value, ast.Dict):
                    module_mappings.add(target.id)
    if not module_mappings:
        return errors
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        if func.attr not in {"update", "setdefault", "pop", "clear"}:
            continue
        value = func.value
        if isinstance(value, ast.Name) and value.id in module_mappings:
            errors.append(
                f"{path}:{node.lineno}: in-place mutation of module-level "
                f"mapping '{value.id}'"
            )
    return errors


def main() -> int:
    all_errors: list[str] = []
    for rel in TARGET_FILES:
        path = REPO_ROOT / rel
        if not path.exists():
            all_errors.append(f"missing target file: {rel}")
            continue
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            all_errors.append(f"{rel}: syntax error: {exc}")
            continue
        all_errors.extend(check_mutable_defaults(path, tree))
        all_errors.extend(check_hasattr_on_dataclass_fields(path, tree))
        all_errors.extend(check_module_level_mapping_mutation(path, tree))

    if all_errors:
        for err in all_errors:
            print(err, file=sys.stderr)
        return 1
    print("static pattern checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
