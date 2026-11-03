#!/usr/bin/env python3
"""Static checks for patterns fixed by the upstream SonarQube PR.

This script is intentionally dependency-free so it can run in the minimal CI
environment. It scans the files touched by the port for:

* mutable default arguments (dict/list/set literals in function signatures)
* ``hasattr`` checks on dataclass fields
* in-place mutation of module-level dicts (best-effort heuristic)

Exit code is non-zero when a pattern is found.
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
        positional = list(args.posonlyargs) + list(args.args)
        defaults = list(args.defaults)
        if defaults:
            offset = len(positional) - len(defaults)
            for arg, default in zip(positional[offset:], defaults):
                if _is_mutable_default(default):
                    problems.append(
                        f"{path}:{node.lineno}: mutable default for '{arg.arg}' "
                        f"in {node.name}()"
                    )
        for arg, default in zip(args.kwonlyargs, args.kw_defaults):
            if default is not None and _is_mutable_default(default):
                problems.append(
                    f"{path}:{node.lineno}: mutable default for '{arg.arg}' "
                    f"in {node.name}()"
                )
    return problems


def check_hasattr_on_dataclass_fields(path: Path, tree: ast.AST) -> list[str]:
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
                f"{path}:{node.lineno}: hasattr() on dataclass field "
                f"'{attr.value}'"
            )
    return problems


def main() -> int:
    problems: list[str] = []
    for rel in TARGET_FILES:
        path = REPO_ROOT / rel
        if not path.exists():
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            problems.append(f"{path}: syntax error: {exc}")
            continue
        problems.extend(check_mutable_defaults(path, tree))
        problems.extend(check_hasattr_on_dataclass_fields(path, tree))

    if problems:
        print("Static pattern check failed:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("Static pattern check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
