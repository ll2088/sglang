#!/usr/bin/env python3
"""Stdlib-only checks for Dockerfile permissions and non-root cache docs.

These checks intentionally avoid importing sglang or pytest so CI can run them
with a bare ``python3`` interpreter, without installing the full sglang
dependency tree. The canonical entry points are:

    python3 test/dockerfile_permissions/run.py
    python3 test/dockerfile_permissions/test_dockerfile_permissions.py
    python3 -m pytest -q -c test/dockerfile_permissions/pytest.ini \
        test/dockerfile_permissions/
"""

import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))

DOCKERFILES = [
    "docker/Dockerfile",
    "examples/frontend_language/usage/triton/Dockerfile",
    "python/sglang/kernels/aot/Dockerfile",
    ".devcontainer/Dockerfile",
]

READMES = [
    "README.md",
    "examples/frontend_language/usage/triton/README.md",
    "python/sglang/kernels/aot/README.md",
]

# Matches `chmod 777`, `chmod -R 777`, `chmod -R 0777`, `chmod a+rwx`, etc.
_WORLD_WRITABLE_RE = re.compile(
    r"chmod\s+(?:-[A-Za-z]+\s+)*(?:0?777|a\+rwx|ugo\+rwx)\b"
)

# Matches `chmod -R go-w` (with optional extra flags before/after).
_GO_W_RE = re.compile(r"chmod\s+(?:-[A-Za-z]+\s+)*go-w\b")

# Matches `chown -R <user>` or `chown -R <user>:<group>`.
_CHOWN_RE = re.compile(r"chown\s+(?:-[A-Za-z]+\s+)*[A-Za-z0-9_.-]+(?::[A-Za-z0-9_.-]+)?\b")

# Keywords that indicate non-root cache guidance in a README.
_README_KEYWORDS = [
    "cache",
    "chown",
    "chmod",
    "go-w",
    "non-root",
    "TORCH_EXTENSIONS_DIR",
    "SGLANG_CACHE_DIR",
]


def _read(rel_path):
    path = os.path.join(_REPO_ROOT, rel_path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _dockerfile_lines(rel_path):
    """Yield (line_number, line) for non-comment Dockerfile lines."""
    for lineno, line in enumerate(_read(rel_path).splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        yield lineno, line


def check_no_world_writable_permissions():
    """No Dockerfile may grant world-writable permissions."""
    failures = []
    for rel_path in DOCKERFILES:
        for lineno, line in _dockerfile_lines(rel_path):
            if _WORLD_WRITABLE_RE.search(line):
                failures.append(f"{rel_path}:{lineno}: world-writable chmod: {line.strip()}")
    assert not failures, "world-writable permissions found:\n" + "\n".join(failures)


def check_minimal_permission_statements():
    """Each Dockerfile must chown a runtime user and drop group/other write."""
    failures = []
    for rel_path in DOCKERFILES:
        text = "\n".join(line for _, line in _dockerfile_lines(rel_path))
        if not _CHOWN_RE.search(text):
            failures.append(f"{rel_path}: missing chown to a runtime user")
        if not _GO_W_RE.search(text):
            failures.append(f"{rel_path}: missing chmod go-w")
    assert not failures, "minimal permission statements missing:\n" + "\n".join(failures)


def check_devcontainer_workspace_not_world_writable():
    """The devcontainer workspace must not be world-writable."""
    rel_path = ".devcontainer/Dockerfile"
    for lineno, line in _dockerfile_lines(rel_path):
        if "/sgl-workspace" in line and _WORLD_WRITABLE_RE.search(line):
            raise AssertionError(
                f"{rel_path}:{lineno}: /sgl-workspace is world-writable: {line.strip()}"
            )


def check_readme_non_root_cache_guidance():
    """Each README must document a writable cache dir for non-root runs."""
    failures = []
    for rel_path in READMES:
        text = _read(rel_path)
        if "non-root" not in text.lower():
            failures.append(f"{rel_path}: missing non-root section")
        if "TORCH_EXTENSIONS_DIR" not in text and "SGLANG_CACHE_DIR" not in text:
            failures.append(f"{rel_path}: missing writable cache env var guidance")
        if "docker run" not in text:
            failures.append(f"{rel_path}: missing executable docker run example")
    assert not failures, "README guidance missing:\n" + "\n".join(failures)


def check_readme_keyword_hit_threshold(min_hits=6):
    """READMEs must mention enough cache/permission keywords in aggregate."""
    hits = 0
    for rel_path in READMES:
        text = _read(rel_path).lower()
        for keyword in _README_KEYWORDS:
            if keyword.lower() in text:
                hits += 1
    assert hits >= min_hits, f"README keyword hits {hits} < {min_hits}"


CHECKS = [
    check_no_world_writable_permissions,
    check_minimal_permission_statements,
    check_devcontainer_workspace_not_world_writable,
    check_readme_non_root_cache_guidance,
    check_readme_keyword_hit_threshold,
]


def main():
    failures = 0
    for check in CHECKS:
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
