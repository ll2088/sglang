#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Static checks for the patterns fixed by the upstream port.

This script is intentionally dependency-free (stdlib only) so it can run in
the minimal CI environment. It scans the files touched by this port for:

1. mutable default arguments (``=[]``, ``={}``, ``=set()``) in function
   signatures;
2. in-place mutation of module-level dicts via ``.update(`` / ``[...] =``
   on a known module-level mapping name;
3. ``hasattr`` checks on dataclass fields (heuristic: ``hasattr(x, "field")``
   where ``field`` is declared in a nearby ``@dataclass``).

Exit code is non-zero if any pattern is found.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Files that this port touched and that must stay clean.
TARGET_FILES = [
    "python/sglang/srt/layers/quantization/utils.py",
    "python/sglang/srt/layers/quantization/compressed_tensors/utils.py",
    "python/sglang/srt/layers/quantization/quark/utils.py",
    "python/sglang/srt/models/inkling_common/quantization/config.py",
    "python/sglang/srt/utils/hf_transformers/config.py",
    "python/sglang/srt/model_loader/utils.py",
]

MUTABLE_DEFAULT_RE = re.compile(
    r"def\s+\w+\s*\([^)]*=\s*(\[\]|\{\}|set\(\))", re.DOTALL
)


def _check_mutable_defaults(path: Path, source: str) -> list[str]:
    errors: list[str] = []
    for match in MUTABLE_DEFAULT_RE.finditer(source):
        line = source[: match.start()].count("\n") + 1
        errors.append(f"{path}:{line}: mutable default argument {match.group(1)!r}")
    return errors


def _module_level_mapping_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and isinstance(
                    node.value, (ast.Dict, ast.Call)
                ):
                    names.add(target.id)
    return names


def _check_inplace_mutation(path: Path, tree: ast.Module) -> list[str]:
    errors: list[str] = []
    module_names = _module_level_mapping_names(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {"update", "setdefault", "pop", "clear"}:
                value = node.func.value
                if isinstance(value, ast.Name) and value.id in module_names:
                    errors.append(
                        f"{path}:{node.lineno}: in-place mutation of module-level "
                        f"mapping {value.id!r} via .{node.func.attr}()"
                    )
    return errors


def _dataclass_field_names(tree: ast.Module) -> set[str]:
    fields: set[str] = set()
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
                    fields.add(stmt.target.id)
    return fields


def _check_hasattr_on_dataclass_fields(path: Path, tree: ast.Module) -> list[str]:
    errors: list[str] = []
    fields = _dataclass_field_names(tree)
    if not fields:
        return errors
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "hasattr"
            and len(node.args) == 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value in fields
        ):
            errors.append(
                f"{path}:{node.lineno}: hasattr() on dataclass field "
                f"{node.args[1].value!r}; use `is not None`"
            )
    return errors


def main() -> int:
    all_errors: list[str] = []
    for rel in TARGET_FILES:
        path = REPO_ROOT / rel
        if not path.exists():
            continue
        source = path.read_text(encoding="utf-8")
        all_errors.extend(_check_mutable_defaults(path, source))
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            all_errors.append(f"{path}: syntax error: {exc}")
            continue
        all_errors.extend(_check_inplace_mutation(path, tree))
        all_errors.extend(_check_hasattr_on_dataclass_fields(path, tree))

    if all_errors:
        print("Static pattern check failed:")
        for err in all_errors:
            print(f"  {err}")
        return 1

    print("Static pattern check passed: no mutable defaults, module-level "
          "in-place mutations, or dataclass hasattr misuse found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
