#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Static checks for the patterns fixed by the upstream port.

This script is intentionally dependency-free so that it can run in the
lightweight CI job before any package installation. It scans the files
that were touched by the port for:

* mutable default arguments (``dict``/``list``/``set`` literals)
* ``hasattr`` checks on dataclass fields
* ``weights_only=False`` pickle fallbacks

Exit code is non-zero when a pattern is found.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Files that are part of the port and must stay clean.
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


def check_mutable_defaults(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:  # pragma: no cover - defensive
        return [f"{path}: syntax error: {exc}"]

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
                        f"argument '{arg.arg}' in '{node.name}'"
                    )
        for arg, default in zip(args.kwonlyargs, args.kw_defaults):
            if default is not None and _is_mutable_default(default):
                errors.append(
                    f"{path}:{node.lineno}: mutable default for "
                    f"keyword argument '{arg.arg}' in '{node.name}'"
                )
    return errors


def check_hasattr_on_dataclass_fields(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:  # pragma: no cover - defensive
        return [f"{path}: syntax error: {exc}"]

    dataclass_fields: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        is_dataclass = any(
            (isinstance(dec, ast.Name) and dec.id == "dataclass")
            or (isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name)
                and dec.func.id == "dataclass")
            for dec in node.decorator_list
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
                f"{path}:{node.lineno}: hasattr on dataclass field "
                f"'{attr.value}' (use 'is not None')"
            )
    return errors


def check_unsafe_pickle(path: Path) -> list[str]:
    errors: list[str] = []
    text = path.read_text(encoding="utf-8")
    for lineno, line in enumerate(text.splitlines(), start=1):
        if "weights_only=False" in line:
            errors.append(
                f"{path}:{lineno}: unsafe pickle fallback 'weights_only=False'"
            )
    return errors


def main() -> int:
    errors: list[str] = []
    for rel in TARGET_FILES:
        path = REPO_ROOT / rel
        if not path.exists():
            continue
        errors.extend(check_mutable_defaults(path))
        errors.extend(check_hasattr_on_dataclass_fields(path))
        errors.extend(check_unsafe_pickle(path))

    if errors:
        for err in errors:
            print(err, file=sys.stderr)
        return 1
    print("static pattern checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
