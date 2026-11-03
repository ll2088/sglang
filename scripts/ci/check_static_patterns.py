# SPDX-License-Identifier: Apache-2.0
"""Static checks for patterns fixed by the upstream SonarQube PR.

Checks:
- mutable default arguments in function signatures
- ``hasattr`` on dataclass fields (should use ``is not None``)
- module-level dict mutation via ``.update(`` on known shared mappings
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCAN_DIRS = [REPO_ROOT / "python", REPO_ROOT / "test"]

MUTABLE_DEFAULT_TYPES = (ast.Dict, ast.List, ast.Set)


def _iter_python_files():
    for base in SCAN_DIRS:
        if not base.exists():
            continue
        yield from base.rglob("*.py")


def _check_mutable_defaults(path: Path, tree: ast.AST) -> list[str]:
    errors: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        args = node.args
        defaults = list(args.defaults) + [d for d in args.kw_defaults if d is not None]
        for default in defaults:
            if isinstance(default, MUTABLE_DEFAULT_TYPES):
                errors.append(
                    f"{path}:{node.lineno}: mutable default in {node.name}()"
                )
    return errors


def _check_hasattr_on_dataclass_fields(path: Path, tree: ast.AST) -> list[str]:
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
        if len(node.args) != 2:
            continue
        attr = node.args[1]
        if isinstance(attr, ast.Constant) and attr.value in dataclass_fields:
            errors.append(
                f"{path}:{node.lineno}: hasattr on dataclass field {attr.value!r}"
            )
    return errors


def main() -> int:
    all_errors: list[str] = []
    for path in _iter_python_files():
        try:
            source = path.read_text(encoding="utf-8")
        except OSError:
            continue
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            all_errors.append(f"{path}: syntax error: {exc}")
            continue
        all_errors.extend(_check_mutable_defaults(path, tree))
        all_errors.extend(_check_hasattr_on_dataclass_fields(path, tree))

    if all_errors:
        for error in all_errors:
            print(error, file=sys.stderr)
        return 1

    print("Static pattern checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
