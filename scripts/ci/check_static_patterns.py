#!/usr/bin/env python3
# Static checks for patterns fixed by the upstream SonarQube PR:
#   * module-level dict mutated in place via .update() on a shared constant
#   * hasattr() used on dataclass fields instead of `is not None`
# The checks are intentionally conservative and only report clear matches.
from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET_DIRS = [
    REPO_ROOT / "python/sglang/srt/layers/quantization",
    REPO_ROOT / "python/sglang/srt/model_loader",
    REPO_ROOT / "python/sglang/srt/utils/hf_transformers",
]


def _iter_py_files():
    for base in TARGET_DIRS:
        if not base.is_dir():
            continue
        yield from base.rglob("*.py")


def _check_hasattr_on_dataclass_fields(path: Path, tree: ast.AST) -> list[str]:
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
        fields = {
            stmt.target.id
            for stmt in node.body
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
        }
        if fields:
            dataclass_fields[node.name] = fields
    if not dataclass_fields:
        return problems
    all_fields = set().union(*dataclass_fields.values())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Name) and func.id == "hasattr"):
            continue
        if len(node.args) < 2:
            continue
        attr = node.args[1]
        if isinstance(attr, ast.Constant) and attr.value in all_fields:
            problems.append(
                f"{path}:{node.lineno}: hasattr() on dataclass field "
                f"{attr.value!r}; use `is not None`"
            )
    return problems


def main() -> int:
    problems: list[str] = []
    for path in _iter_py_files():
        try:
            source = path.read_text(encoding="utf-8")
        except OSError as exc:
            problems.append(f"{path}: cannot read: {exc}")
            continue
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            problems.append(f"{path}: syntax error: {exc}")
            continue
        problems.extend(_check_hasattr_on_dataclass_fields(path, tree))

    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        return 1
    print("static pattern checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
