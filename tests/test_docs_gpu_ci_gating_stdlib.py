"""Dependency-free documentation consistency checks for GPU CI gating.

Runs with the standard library only (``python tests/test_docs_gpu_ci_gating_stdlib.py``)
so it can execute in environments where pytest is not installed. The pytest
counterpart lives in ``tests/test_docs_gpu_ci_gating.py``.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS = [
    ROOT / "docs" / "CONTRIBUTING.md",
    ROOT / ".agents" / "skills" / "ci-workflow-guide" / "SKILL.md",
    ROOT / ".agents" / "skills" / "babysit-pr-to-pass-ci" / "SKILL.md",
]


def _read(path):
    return path.read_text(encoding="utf-8")


def check_mentions_gpu_ci():
    failures = []
    for path in DOCS:
        if "GPU CI" not in _read(path):
            failures.append(f"{path} must mention GPU CI")
    return failures


def check_main_target_requirement():
    failures = []
    for path in DOCS:
        text = _read(path).lower()
        if not ("target" in text and "main" in text):
            failures.append(
                f"{path} must state that GPU CI runs only for PRs targeting main"
            )
    return failures


def check_non_main_skip():
    failures = []
    for path in DOCS:
        if "skip" not in _read(path).lower():
            failures.append(f"{path} must state that non-main PRs skip GPU CI")
    return failures


def check_stacked_and_merge_base():
    failures = []
    for path in DOCS:
        text = _read(path).lower()
        if "stacked" not in text:
            failures.append(f"{path} must explain stacked PRs")
        if "merge-base" not in text:
            failures.append(f"{path} must explain merge-base detection")
    return failures


def check_retarget_requires_push_or_rerun():
    failures = []
    for path in DOCS:
        text = _read(path).lower()
        if "retarget" not in text:
            failures.append(
                f"{path} must state that retargeting to main needs a new push or full re-run"
            )
    return failures


CHECKS = [
    check_mentions_gpu_ci,
    check_main_target_requirement,
    check_non_main_skip,
    check_stacked_and_merge_base,
    check_retarget_requires_push_or_rerun,
]


def main():
    failures = []
    for check in CHECKS:
        failures.extend(check())
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    print(f"OK: {len(CHECKS)} documentation checks passed for {len(DOCS)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
