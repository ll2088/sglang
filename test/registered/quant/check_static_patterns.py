# SPDX-License-Identifier: Apache-2.0
"""Static checks for the patterns fixed by the upstream port.

This script is intentionally dependency-free (stdlib only) so it can run in
the minimal CI environment. It scans the files touched by the port for:

* mutable default arguments (dict/list/set literals in signatures)
* in-place mutation of module-level dicts
* ``hasattr`` checks on dataclass fields

Exit code is non-zero when a pattern is found.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

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
        defaults = list(args.defaults) + [d for d in args.kw_defaults if d is not None]
        for default in defaults:
            if _is_mutable_default(default):
                problems.append(
                    f"{path}:{node.lineno}: mutable default in {node.name}()"
                )
    return problems


def _check_module_dict_mutation(tree: ast.AST, path: Path) -> list[str]:
    problems: list[str] = []
    module_level_names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    module_level_names.add(target.id)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        if func.attr not in {"update", "setdefault", "pop", "clear"}:
            continue
        value = func.value
        if isinstance(value, ast.Name) and value.id in module_level_names:
            problems.append(
                f"{path}:{node.lineno}: in-place mutation of module-level {value.id}"
            )
    return problems


def _check_hasattr_on_dataclass(tree: ast.AST, path: Path) -> list[str]:
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
    for rel in TARGET_FILES:
        path = REPO_ROOT / rel
        if not path.exists():
            continue
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        problems.extend(_check_mutable_defaults(tree, path))
        problems.extend(_check_module_dict_mutation(tree, path))
        problems.extend(_check_hasattr_on_dataclass(tree, path))
    if problems:
        for problem in problems:
            print(problem)
        return 1
    print("static pattern checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
