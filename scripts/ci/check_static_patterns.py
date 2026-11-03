#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Static checks for patterns fixed by the upstream port.

Checks:
- mutable default arguments (dict/list/set literals) in function signatures
- ``hasattr`` on dataclass fields (should be ``is not None``)

Exits non-zero when a violation is found so CI can block the job.
"""

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET_DIRS = [ROOT / "python" / "sglang"]

MUTABLE_NODES = (ast.Dict, ast.List, ast.Set, ast.DictComp, ast.ListComp, ast.SetComp)


def _iter_py_files():
    for base in TARGET_DIRS:
        if not base.exists():
            continue
        yield from base.rglob("*.py")


def _check_mutable_defaults(path, tree, violations):
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        args = node.args
        all_args = list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)
        defaults = list(args.defaults) + [d for d in args.kw_defaults if d is not None]
        for arg, default in zip(all_args[-len(defaults):] if defaults else [], defaults):
            if isinstance(default, MUTABLE_NODES):
                violations.append(
                    f"{path}:{node.lineno}: mutable default for '{arg.arg}' in {node.name}()"
                )


def _check_dataclass_hasattr(path, tree, violations):
    dataclass_fields = set()
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
        return
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
            violations.append(
                f"{path}:{node.lineno}: hasattr on dataclass field '{attr.value}'"
            )


def main():
    violations = []
    for path in _iter_py_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            violations.append(f"{path}: syntax error: {exc}")
            continue
        _check_mutable_defaults(path, tree, violations)
        _check_dataclass_hasattr(path, tree, violations)
    if violations:
        print("Static pattern violations found:")
        for v in violations:
            print(f"  {v}")
        return 1
    print("No static pattern violations found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
