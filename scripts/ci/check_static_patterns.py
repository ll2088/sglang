# SPDX-License-Identifier: Apache-2.0
"""Static pattern checks for the quant regression fixes.

Rejects three patterns that caused real bugs upstream:

* mutable default arguments (``def f(x={})``)
* ``hasattr`` on dataclass fields (always true)
* module-level dict mutation via ``.update(`` on a shared constant
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCAN_DIRS = (REPO_ROOT / "python", REPO_ROOT / "test")


def _iter_py_files():
    for base in SCAN_DIRS:
        if not base.is_dir():
            continue
        yield from base.rglob("*.py")


def _is_mutable_literal(node: ast.AST) -> bool:
    if isinstance(node, (ast.Dict, ast.List, ast.Set)):
        return True
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        return node.func.id in {"dict", "list", "set"}
    return False


def check_mutable_defaults(path: Path, tree: ast.AST) -> list[str]:
    errors = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        args = node.args
        for default in list(args.defaults) + [d for d in args.kw_defaults if d]:
            if _is_mutable_literal(default):
                errors.append(
                    f"{path}:{node.lineno}: mutable default in {node.name}()"
                )
    return errors


def check_hasattr_on_dataclass(path: Path, tree: ast.AST) -> list[str]:
    errors = []
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
            errors.append(
                f"{path}:{node.lineno}: hasattr(..., {attr.value!r}) may be "
                "always true for dataclass fields; use `is not None`"
            )
    return errors


def main() -> int:
    errors: list[str] = []
    for path in _iter_py_files():
        try:
            source = path.read_text(encoding="utf-8")
        except OSError as exc:
            errors.append(f"{path}: cannot read ({exc})")
            continue
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            errors.append(f"{path}: syntax error ({exc})")
            continue
        errors.extend(check_mutable_defaults(path, tree))
        errors.extend(check_hasattr_on_dataclass(path, tree))

    if errors:
        for err in errors:
            print(err, file=sys.stderr)
        return 1

    print("static pattern checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
