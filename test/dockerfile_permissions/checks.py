#!/usr/bin/env python3
"""Stdlib-only checks for Dockerfile permission hardening and README guidance.

These checks intentionally avoid importing sglang or pytest so CI can run them
with a bare python3 interpreter. They are also re-exported by
``test_dockerfile_permissions.py`` for pytest collection.

Run standalone with:
    python3 test/dockerfile_permissions/checks.py
"""

import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))

# Dockerfiles that ship code loaded into a serving/build process.
DOCKERFILES = [
    "docker/Dockerfile",
    "examples/frontend_language/usage/triton/Dockerfile",
    "python/sglang/kernels/aot/Dockerfile",
    ".devcontainer/Dockerfile",
]

# READMEs that must document how to run as a non-default user.
READMES = [
    "README.md",
    "examples/frontend_language/usage/triton/README.md",
    "python/sglang/kernels/aot/README.md",
]

# Matches `chmod 777`, `chmod -R 777`, `chmod -R 0777`, etc.
_WORLD_WRITABLE_RE = re.compile(r"chmod\s+(?:-R\s+)?0?777\b")

# Matches a chown to a non-root user (e.g. `chown -R vllm /workspace`).
_CHOWN_USER_RE = re.compile(r"chown\s+(?:-R\s+)?[A-Za-z_][\w.-]*\s")

# Matches `chmod -R go-w` (with optional extra flags before the mode).
_GO_W_RE = re.compile(r"chmod\s+(?:-R\s+)?(?:-[A-Za-z]+\s+)*go-w\b")

# Keywords that indicate non-root cache guidance in a README.
_README_KEYWORDS = (
    "cache",
    "chown",
    "chmod",
    "go-w",
    "non-root",
    "TORCH_EXTENSIONS_DIR",
    "SGLANG_CACHE_DIR",
)


def _read(rel_path):
    path = os.path.join(_REPO_ROOT, rel_path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def check_no_world_writable_permissions():
    """No affected Dockerfile may grant world-writable permissions."""
    offenders = []
    for rel in DOCKERFILES:
        text = _read(rel)
        for lineno, line in enumerate(text.splitlines(), 1):
            if _WORLD_WRITABLE_RE.search(line):
                offenders.append(f"{rel}:{lineno}: {line.strip()}")
    assert not offenders, "world-writable chmod found:\n" + "\n".join(offenders)


def check_minimal_permission_statements():
    """Each affected Dockerfile must chown a user and drop group/other write."""
    missing = []
    for rel in DOCKERFILES:
        text = _read(rel)
        if not _CHOWN_USER_RE.search(text):
            missing.append(f"{rel}: missing chown to a non-root user")
        if not _GO_W_RE.search(text):
            missing.append(f"{rel}: missing chmod go-w")
    assert not missing, "missing minimal-permission statements:\n" + "\n".join(
        missing
    )


def check_devcontainer_workspace_not_world_writable():
    """The devcontainer workspace must not be world-writable."""
    text = _read(".devcontainer/Dockerfile")
    assert not _WORLD_WRITABLE_RE.search(text), (
        ".devcontainer/Dockerfile grants world-writable permissions"
    )


def check_readme_non_root_cache_guidance():
    """Each README must document a writable cache dir for non-root users."""
    missing = []
    for rel in READMES:
        text = _read(rel)
        lowered = text.lower()
        if "non-root" not in lowered and "non-default user" not in lowered:
            missing.append(f"{rel}: missing non-root section")
        if "docker run" not in text:
            missing.append(f"{rel}: missing executable docker run example")
        if "TORCH_EXTENSIONS_DIR" not in text and "SGLANG_CACHE_DIR" not in text:
            missing.append(f"{rel}: missing cache directory env var")
    assert not missing, "missing README guidance:\n" + "\n".join(missing)


def check_readme_keyword_hit_threshold(min_hits=6):
    """READMEs must collectively mention enough cache/permission keywords."""
    hits = 0
    for rel in READMES:
        text = _read(rel)
        lowered = text.lower()
        for kw in _README_KEYWORDS:
            hits += lowered.count(kw.lower())
    assert hits >= min_hits, f"README keyword hits {hits} < {min_hits}"


CHECKS = (
    check_no_world_writable_permissions,
    check_minimal_permission_statements,
    check_devcontainer_workspace_not_world_writable,
    check_readme_non_root_cache_guidance,
    check_readme_keyword_hit_threshold,
)


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
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
