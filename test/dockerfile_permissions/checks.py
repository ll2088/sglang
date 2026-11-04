#!/usr/bin/env python3
"""Stdlib-only checks for Dockerfile permission hardening and README guidance.

These checks intentionally avoid importing sglang, pytest, or any third-party
package so they can run under a bare ``python3`` interpreter in CI without
installing the full sglang dependency tree.

Usage:
    python3 test/dockerfile_permissions/checks.py
    python3 test/dockerfile_permissions/run.py
"""

import os
import re
import sys

# Repository root is two levels up from this file (test/dockerfile_permissions/).
_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))

# Dockerfiles that must not grant world-writable access to runtime directories.
DOCKERFILES = [
    "docker/Dockerfile",
    "examples/frontend_language/usage/triton/Dockerfile",
    "python/sglang/kernels/aot/Dockerfile",
    ".devcontainer/Dockerfile",
]

# READMEs that must document how to run as a non-default user with a writable
# cache directory after the permission tightening.
READMES = [
    "README.md",
    "examples/frontend_language/usage/triton/README.md",
    "python/sglang/kernels/aot/README.md",
]

# Matches `chmod 777`, `chmod -R 777`, `chmod -R 0777`, `chmod a+rwx`, etc.
_WORLD_WRITABLE_RE = re.compile(
    r"chmod\s+(?:-R\s+)?(?:0?777|a\+rwx|ugo\+rwx|o\+w)\b",
    re.IGNORECASE,
)

# Matches a chown to a non-root user (e.g. `chown -R vllm /workspace`).
_CHOWN_USER_RE = re.compile(r"chown\s+(?:-R\s+)?[A-Za-z0-9_.-]+\s+", re.IGNORECASE)

# Matches `chmod -R go-w` or `chmod go-w` (group/other write removal).
_CHMOD_GO_W_RE = re.compile(r"chmod\s+(?:-R\s+)?go-w\b", re.IGNORECASE)

# README keywords that indicate non-root cache guidance is present.
_README_KEYWORDS = [
    "cache",
    "chown",
    "chmod",
    "go-w",
    "non-root",
    "TORCH_EXTENSIONS_DIR",
    "SGLANG_CACHE_DIR",
]


def _read(path):
    full = os.path.join(REPO_ROOT, path)
    with open(full, "r", encoding="utf-8") as fh:
        return fh.read()


def check_no_world_writable_permissions():
    """No Dockerfile may grant world-writable permissions."""
    failures = []
    for path in DOCKERFILES:
        text = _read(path)
        for lineno, line in enumerate(text.splitlines(), 1):
            if _WORLD_WRITABLE_RE.search(line):
                failures.append(f"{path}:{lineno}: world-writable chmod: {line.strip()}")
    assert not failures, "world-writable permissions found:\n" + "\n".join(failures)


def check_minimal_permission_statements():
    """Each Dockerfile must chown a runtime user and remove group/other write."""
    failures = []
    for path in DOCKERFILES:
        text = _read(path)
        if not _CHOWN_USER_RE.search(text):
            failures.append(f"{path}: missing chown to a runtime user")
        if not _CHMOD_GO_W_RE.search(text):
            failures.append(f"{path}: missing chmod go-w")
    assert not failures, "missing minimal-permission statements:\n" + "\n".join(failures)


def check_devcontainer_workspace_not_world_writable():
    """.devcontainer must not leave /sgl-workspace world-writable."""
    text = _read(".devcontainer/Dockerfile")
    for lineno, line in enumerate(text.splitlines(), 1):
        if _WORLD_WRITABLE_RE.search(line):
            raise AssertionError(
                f".devcontainer/Dockerfile:{lineno}: world-writable chmod: {line.strip()}"
            )


def check_readme_non_root_cache_guidance():
    """Each README must document a writable cache dir for non-root runs."""
    failures = []
    for path in READMES:
        text = _read(path)
        lowered = text.lower()
        hits = sum(1 for kw in _README_KEYWORDS if kw.lower() in lowered)
        if hits < 3:
            failures.append(
                f"{path}: only {hits} non-root cache keywords found (need >= 3)"
            )
        if "docker run" not in text:
            failures.append(f"{path}: missing an executable `docker run` example")
    assert not failures, "README guidance missing:\n" + "\n".join(failures)


def check_readme_keyword_hit_threshold():
    """Aggregate README keyword hits must be at least 6."""
    total = 0
    for path in READMES:
        lowered = _read(path).lower()
        total += sum(1 for kw in _README_KEYWORDS if kw.lower() in lowered)
    assert total >= 6, f"README keyword hits too low: {total} < 6"


ALL_CHECKS = [
    check_no_world_writable_permissions,
    check_minimal_permission_statements,
    check_devcontainer_workspace_not_world_writable,
    check_readme_non_root_cache_guidance,
    check_readme_keyword_hit_threshold,
]


def main():
    failures = 0
    for check in ALL_CHECKS:
        try:
            check()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {check.__name__}: {exc}", file=sys.stderr)
        else:
            print(f"PASS {check.__name__}")
    if failures:
        print(f"{failures} check(s) failed", file=sys.stderr)
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
