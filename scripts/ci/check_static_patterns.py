# SPDX-License-Identifier: Apache-2.0
"""Static review helper for the patterns fixed by the upstream port.

Checks the files touched by this change for:
  * mutable default arguments (dict/list/set literals in signatures)
  * in-place mutation of module-level dicts
  * ``hasattr`` used on dataclass fields

Exits non-zero when a pattern is found so CI can surface it as evidence.
Only the standard library is used.
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


def check_mutable_defaults(tree: ast.AST, path: Path) -> list[str]:
    findings: list[str] = []
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
                    findings.append(
                        f"{path}:{node.lineno}: mutable default for '{arg.arg}' "
                        f"in {node.name}()"
                    )
        for arg, default in zip(args.kwonlyargs, args.kw_defaults):
            if default is not None and _is_mutable_default(default):
                findings.append(
                    f"{path}:{node.lineno}: mutable default for '{arg.arg}' "
                    f"in {node.name}()"
                )
    return findings


def check_module_dict_mutation(tree: ast.AST, path: Path) -> list[str]:
    findings: list[str] = []
    module_level_names = {
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
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
            findings.append(
                f"{path}:{node.lineno}: in-place mutation of module-level "
                f"'{value.id}.{func.attr}()'"
            )
    return findings


def check_hasattr_on_dataclass(tree: ast.AST, path: Path) -> list[str]:
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
    findings: list[str] = []
    if not dataclass_fields:
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
        if isinstance(attr, ast.Constant) and attr.value in dataclass_fields:
            findings.append(
                f"{path}:{node.lineno}: hasattr() on dataclass field "
                f"'{attr.value}'"
            )
    return findings


def main() -> int:
    findings: list[str] = []
    for rel in TARGET_FILES:
        path = REPO_ROOT / rel
        if not path.is_file():
            print(f"skip (missing): {rel}")
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        findings.extend(check_mutable_defaults(tree, path))
        findings.extend(check_module_dict_mutation(tree, path))
        findings.extend(check_hasattr_on_dataclass(tree, path))

    if findings:
        print("static pattern findings:")
        for item in findings:
            print(f"  - {item}")
        return 1
    print("no mutable-default / module-dict-mutation / hasattr findings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
