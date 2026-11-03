#!/usr/bin/env python3
"""Static checks for patterns targeted by the upstream port.

Fails (exit 1) when any of the following is found in the scanned files:
  * mutable default arguments (dict/list/set literals in signatures)
  * ``hasattr`` used on dataclass optional fields
  * ``weights_only=False`` in torch.load calls

This script is intentionally dependency-free so it can run in the minimal
CI environment.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCAN_DIRS = [
    ROOT / "python" / "sglang" / "srt" / "layers" / "quantization",
    ROOT / "python" / "sglang" / "srt" / "model_loader",
    ROOT / "python" / "sglang" / "srt" / "utils" / "hf_transformers",
    ROOT / "python" / "sglang" / "srt" / "models" / "inkling_common",
]

MUTABLE_DEFAULT_RE = re.compile(r"=\s*(\{\}|\[\]|set\(\))")
WEIGHTS_ONLY_FALSE_RE = re.compile(r"weights_only\s*=\s*False")
HASATTR_DATACLASS_RE = re.compile(r"hasattr\(\s*[\w.]+\s*,\s*['\"][\w]+['\"]\s*\)")


def _iter_py_files():
    for base in SCAN_DIRS:
        if not base.exists():
            continue
        yield from base.rglob("*.py")


def _check_mutable_defaults(path: Path, source: str, errors: list[str]) -> None:
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        errors.append(f"{path}: syntax error: {exc}")
        return
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        args = node.args
        all_defaults = list(args.defaults) + [d for d in args.kw_defaults if d is not None]
        for default in all_defaults:
            if isinstance(default, (ast.Dict, ast.List, ast.Set)):
                errors.append(
                    f"{path}:{node.lineno}: mutable default in {node.name}()"
                )


def _check_regex_patterns(path: Path, source: str, errors: list[str]) -> None:
    for lineno, line in enumerate(source.splitlines(), start=1):
        if WEIGHTS_ONLY_FALSE_RE.search(line):
            errors.append(f"{path}:{lineno}: weights_only=False found")
        if HASATTR_DATACLASS_RE.search(line) and "dataclass" in source.lower():
            errors.append(f"{path}:{lineno}: hasattr on dataclass field")


def main() -> int:
    errors: list[str] = []
    for path in _iter_py_files():
        try:
            source = path.read_text(encoding="utf-8")
        except OSError as exc:
            errors.append(f"{path}: cannot read: {exc}")
            continue
        _check_mutable_defaults(path, source, errors)
        _check_regex_patterns(path, source, errors)

    if errors:
        print("Static pattern check failed:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1
    print("Static pattern check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
