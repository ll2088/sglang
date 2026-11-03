#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Static checks for patterns fixed by the upstream SonarQube port.

This script is intentionally dependency-free so it can run in a minimal CI
environment. It scans the files touched by the port for:

* mutable default arguments (``dict``/``list``/``set`` literals in signatures)
* ``hasattr`` checks on dataclass optional fields
* in-place mutation of module-level fallback mappings
"""

import ast
import sys
from pathlib import Path


TARGET_FILES = [
    "python/sglang/srt/layers/quantization/utils.py",
    "python/sglang/srt/layers/quantization/compressed_tensors/utils.py",
    "python/sglang/srt/layers/quantization/quark/utils.py",
    "python/sglang/srt/models/inkling_common/quantization/config.py",
    "python/sglang/srt/utils/hf_transformers/config.py",
    "python/sglang/srt/model_loader/utils.py",
]


def _is_mutable_literal(node):
    return isinstance(node, (ast.Dict, ast.List, ast.Set))


def check_mutable_defaults(path: Path):
    errors = []
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        args = node.args
        for default in list(args.defaults) + [d for d in args.kw_defaults if d]:
            if _is_mutable_literal(default):
                errors.append(
                    f"{path}:{node.lineno}: mutable default in {node.name}"
                )
    return errors


def check_hasattr_on_dataclass_fields(path: Path):
    errors = []
    tree = ast.parse(path.read_text(), filename=str(path))
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
            if attr.value in {"vocab_embedding", "lm_head"}:
                errors.append(
                    f"{path}:{node.lineno}: hasattr on dataclass field "
                    f"{attr.value!r}; use 'is not None'"
                )
    return errors


def check_module_level_mapping_mutation(path: Path):
    errors = []
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Subscript):
                continue
            value = target.value
            if isinstance(value, ast.Name) and value.id.isupper():
                errors.append(
                    f"{path}:{node.lineno}: in-place mutation of module-level "
                    f"mapping {value.id}"
                )
    return errors


def main():
    root = Path(__file__).resolve().parents[2]
    errors = []
    for rel in TARGET_FILES:
        path = root / rel
        if not path.exists():
            continue
        errors.extend(check_mutable_defaults(path))
        errors.extend(check_hasattr_on_dataclass_fields(path))
        errors.extend(check_module_level_mapping_mutation(path))
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("static pattern checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
