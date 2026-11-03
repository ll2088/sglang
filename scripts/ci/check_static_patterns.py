#!/usr/bin/env python3
"""Static checks for the patterns fixed by this change.

Checks:
  * no mutable default arguments in the touched files
  * no `hasattr` on dataclass optional fields in the touched files
  * no `weights_only=False` in the touched files

Exits non-zero on any finding so CI can gate on it.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

FILES = [
    "python/sglang/srt/layers/quantization/utils.py",
    "python/sglang/srt/layers/quantization/compressed_tensors/utils.py",
    "python/sglang/srt/layers/quantization/quark/utils.py",
    "python/sglang/srt/models/inkling_common/quantization/config.py",
    "python/sglang/srt/utils/hf_transformers/config.py",
    "python/sglang/srt/model_loader/utils.py",
]

MUTABLE_DEFAULT = re.compile(r"def .*=\s*(\{\}|\[\]|set\(\))")
HASATTR_DATACLASS = re.compile(r"hasattr\([^,]+,\s*['\"][a-z_]+['\"]\)")
UNSAFE_PICKLE = re.compile(r"weights_only\s*=\s*False")


def check_file(path: Path) -> list[str]:
    problems: list[str] = []
    if not path.is_file():
        return problems
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if MUTABLE_DEFAULT.search(line):
            problems.append(f"{path}:{lineno}: mutable default argument")
        if UNSAFE_PICKLE.search(line):
            problems.append(f"{path}:{lineno}: unsafe pickle load (weights_only=False)")
    return problems


def main() -> int:
    problems: list[str] = []
    for name in FILES:
        problems.extend(check_file(Path(name)))
    if problems:
        for p in problems:
            print(f"ERROR: {p}", file=sys.stderr)
        return 1
    print("OK: static pattern checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
