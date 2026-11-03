# SPDX-License-Identifier: Apache-2.0
"""Static checks for the patterns fixed by the upstream PR.

Run directly: `python test/registered/quant/check_static_patterns.py`.
Exits non-zero when a forbidden pattern is found in the scanned files.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_TARGETS = [
    _ROOT / "python/sglang/srt/layers/quantization/utils.py",
    _ROOT / "python/sglang/srt/layers/quantization/compressed_tensors/utils.py",
    _ROOT / "python/sglang/srt/layers/quantization/quark/utils.py",
]

_MUTABLE_DEFAULT = re.compile(
    r"def\s+\w+\s*\([^)]*=\s*(\{\}|\[\]|set\(\))", re.DOTALL
)
_HASATTR_DATACLASS = re.compile(r"hasattr\(\s*\w+\s*,\s*['\"]\w+['\"]\s*\)")


def _scan(path: Path) -> list[str]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    problems = []
    for match in _MUTABLE_DEFAULT.finditer(text):
        line = text.count("\n", 0, match.start()) + 1
        problems.append(f"{path}:{line}: mutable default argument")
    return problems


def main() -> int:
    problems: list[str] = []
    for target in _TARGETS:
        problems.extend(_scan(target))
    if problems:
        print("Static pattern check failed:")
        for problem in problems:
            print(f"  {problem}")
        return 1
    print("Static pattern check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
