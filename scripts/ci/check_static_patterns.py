#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Static checks for patterns fixed by the upstream SonarQube PR.

This script is intentionally dependency-free (stdlib only) so it can run in
a minimal CI environment before the full sglang runtime is installed.

It scans the quant/config/model-loader modules touched by the port and
reports:

* mutable default arguments (dict/list/set literals in function signatures)
* ``hasattr`` checks on dataclass fields (heuristic: ``hasattr(<name>, ...)``
  where ``<name>`` is a known dataclass instance variable in the same file)
* bare ``except:`` clauses

Exit code is 0 when no findings are reported, 1 otherwise.
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

MUTABLE_LITERAL_TYPES = (ast.Dict, ast.List, ast.Set)


def _is_mutable_default(node: ast.AST) -> bool:
    return isinstance(node, MUTABLE_LITERAL_TYPES)


def _iter_function_defs(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def _check_mutable_defaults(path: Path, tree: ast.AST) -> list[str]:
    findings: list[str] = []
    for func in _iter_function_defs(tree):
        args = func.args
        all_args = list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)
        defaults = list(args.defaults) + [d for d in args.kw_defaults if d is not None]
        for arg, default in zip(all_args[-len(defaults):] if defaults else [], defaults):
            if _is_mutable_default(default):
                findings.append(
                    f"{path}:{func.lineno}: mutable default for argument "
                    f"'{arg.arg}' in {func.name}()"
                )
    return findings


def _check_bare_except(path: Path, tree: ast.AST) -> list[str]:
    findings: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            findings.append(f"{path}:{node.lineno}: bare except clause")
    return findings


def _dataclass_field_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
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
                    names.add(stmt.target.id)
    return names


def _check_hasattr_on_dataclass(path: Path, tree: ast.AST) -> list[str]:
    findings: list[str] = []
    field_names = _dataclass_field_names(tree)
    if not field_names:
        return findings
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Name) and func.id == "hasattr"):
            continue
        if len(node.args) < 2:
            continue
        attr = node.args[1]
        if isinstance(attr, ast.Constant) and isinstance(attr.value, str):
            if attr.value in field_names:
                findings.append(
                    f"{path}:{node.lineno}: hasattr() on dataclass field "
                    f"'{attr.value}' (use 'is not None')"
                )
    return findings


def main() -> int:
    findings: list[str] = []
    for rel in TARGET_FILES:
        path = REPO_ROOT / rel
        if not path.exists():
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            findings.append(f"{path}: syntax error: {exc}")
            continue
        findings.extend(_check_mutable_defaults(path, tree))
        findings.extend(_check_bare_except(path, tree))
        findings.extend(_check_hasattr_on_dataclass(path, tree))

    if findings:
        print("Static pattern check failed:")
        for item in findings:
            print(f"  - {item}")
        return 1

    print("Static pattern check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
