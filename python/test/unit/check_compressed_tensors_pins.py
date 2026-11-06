"""Lightweight pin consistency check for compressed-tensors.

Runs without importing sglang or torch so it can be used as a fast CI gate
before the heavier environment bootstrap. Exits non-zero when any install
variant disagrees on the pin or a stale 0.18.0 entry remains.
"""

import re
import sys
from pathlib import Path

PYTHON_DIR = Path(__file__).resolve().parents[2]
EXPECTED = "0.20.1"
VARIANT_FILES = [
    "pyproject.toml",
    "pyproject_cpu.toml",
    "pyproject_npu.toml",
    "pyproject_xpu.toml",
    "pyproject_other.toml",
]
# setup.py does not declare the pin itself, but it is part of the install
# surface; keep it in the scan so a future edit cannot reintroduce 0.18.0.
EXTRA_FILES = ["setup.py"]
PIN_RE = re.compile(r'"compressed-tensors==([0-9][^"]*)"')


def _line_numbers(text, needle):
    return [
        index
        for index, line in enumerate(text.splitlines(), start=1)
        if needle in line
    ]


def main():
    failures = []
    for name in VARIANT_FILES:
        path = PYTHON_DIR / name
        if not path.is_file():
            failures.append(f"{name}: missing")
            continue
        text = path.read_text(encoding="utf-8")
        pins = set(PIN_RE.findall(text))
        if pins != {EXPECTED}:
            failures.append(f"{name}: pins {sorted(pins)} != ['{EXPECTED}']")
        for line_no in _line_numbers(text, "compressed-tensors==0.18.0"):
            failures.append(f"{name}:{line_no}: stale 0.18.0 pin")
    for path in PYTHON_DIR.glob("pyproject*.toml"):
        text = path.read_text(encoding="utf-8")
        for line_no in _line_numbers(text, "compressed-tensors==0.18.0"):
            failures.append(f"{path.name}:{line_no}: stale 0.18.0 pin")
    for name in EXTRA_FILES:
        path = PYTHON_DIR / name
        if not path.is_file():
            failures.append(f"{name}: missing")
            continue
        text = path.read_text(encoding="utf-8")
        for line_no in _line_numbers(text, "compressed-tensors==0.18.0"):
            failures.append(f"{name}:{line_no}: stale 0.18.0 pin")
        pins = set(PIN_RE.findall(text))
        if pins and pins != {EXPECTED}:
            failures.append(f"{name}: pins {sorted(pins)} != ['{EXPECTED}']")
        # setup.py does not declare the pin, but it must not hard-code a
        # different compressed-tensors version either.
        for line_no in _line_numbers(text, "compressed-tensors"):
            if "compressed-tensors==" in text.splitlines()[line_no - 1]:
                failures.append(
                    f"{name}:{line_no}: unexpected compressed-tensors pin in setup.py"
                )
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    print(f"ok: all variants pin compressed-tensors=={EXPECTED}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
