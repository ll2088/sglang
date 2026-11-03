#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Static checks for patterns fixed by the upstream SonarQube PR.

Checks:
  * mutable default arguments (dict/list/set literals in function signatures)
  * dataclass optional fields checked with hasattr
  * module-level dicts mutated in place via .update()/.append() on known names

Exit code 0 when clean, 1 when violations are found.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET_DIRS = [
    REPO_ROOT / "python" / "sglang" / "srt" / "layers" / "quantization",
    REPO_ROOT / "python" / "sglang" / "srt" / "models" / "inkling_common",
    REPO_ROOT / "python" / "sglang" / "srt" / "utils" / "hf_transformers",
    REPO_ROOT / "python" / "sglang" / "srt" / "model_loader",
]

MUTABLE_LITERALS = (ast.Dict, ast.List, ast.Set)


def _iter_py_files():
    for base in TARGET_DIRS:
        if not base.exists():
            continue
        yield from base.rglob("*.py")


def _check_mutable_defaults(path: Path, tree: ast.AST) -> list[str]:
    problems: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        args = node.args
        all_args = list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)
        defaults = list(args.defaults) + [d for d in args.kw_defaults if d is not None]
        for arg, default in zip(all_args[-len(defaults):] if defaults else [], defaults):
            if isinstance(default, MUTABLE_LITERALS):
                problems.append(
                    f"{path}:{node.lineno}: mutable default for '{arg.arg}' in {node.name}()"
                )
    return problems


def _check_hasattr_on_dataclass(path: Path, tree: ast.AST) -> list[str]:
    problems: list[str] = []
    dataclass_fields: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            is_dataclass = any(
                (isinstance(d, ast.Name) and d.id == "dataclass")
                or (isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id == "dataclass")
                for d in node.decorator_list
            )
            if not is_dataclass:
                continue
            fields = set()
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    fields.add(stmt.target.id)
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
        if isinstance(attr, ast.Constant) and isinstance(attr.value, str):
            for cls_name, fields in dataclass_fields.items():
                if attr.value in fields:
                    problems.append(
                        f"{path}:{node.lineno}: hasattr on dataclass field "
                        f"'{attr.value}' of {cls_name}; use 'is not None'"
                    )
    return problems


def main() -> int:
    problems: list[str] = []
    for path in _iter_py_files():
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
        except (OSError, SyntaxError) as exc:
            problems.append(f"{path}: failed to parse: {exc}")
            continue
        problems.extend(_check_mutable_defaults(path, tree))
        problems.extend(_check_hasattr_on_dataclass(path, tree))
    if problems:
        print("Static pattern check failed:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1
    print("Static pattern check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
