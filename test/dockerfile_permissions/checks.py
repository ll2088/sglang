#!/usr/bin/env python3
"""Stdlib-only checks for Dockerfile permission hardening and README guidance.

These checks intentionally avoid importing sglang or pytest so CI can run them
with a bare python3 interpreter before any dependency installation. They are
re-exported by test_dockerfile_permissions.py for pytest collection and can be
run directly via run.py.
"""

import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

# Matches chmod with an optional -R flag followed by 777 (world-writable).
WORLD_WRITABLE_RE = re.compile(r"chmod\s+(?:-R\s+)?777\b")

# Matches a chown statement (any target) and a chmod go-w statement.
CHOWN_RE = re.compile(r"\bchown\s+-R\s+\S+")
GO_W_RE = re.compile(r"chmod\s+-R\s+go-w\b")

# Keywords that indicate non-root cache guidance in a README.
README_KEYWORDS = [
    "cache",
    "chown",
    "chmod",
    "go-w",
    "non-root",
    "TORCH_EXTENSIONS_DIR",
]


def _read(rel_path):
    path = os.path.join(REPO_ROOT, rel_path)
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def check_no_world_writable_permissions():
    """No affected Dockerfile may grant world-writable permissions."""
    offenders = []
    for rel_path in DOCKERFILES:
        text = _read(rel_path)
        for match in WORLD_WRITABLE_RE.finditer(text):
            line_no = text.count("\n", 0, match.start()) + 1
            offenders.append(f"{rel_path}:{line_no}: {match.group(0)}")
    assert not offenders, "world-writable chmod found:\n" + "\n".join(offenders)


def check_minimal_permission_statements():
    """Affected Dockerfiles must use chown + chmod go-w instead of 777."""
    missing = []
    for rel_path in DOCKERFILES:
        text = _read(rel_path)
        if not CHOWN_RE.search(text) or not GO_W_RE.search(text):
            missing.append(rel_path)
    assert not missing, "missing chown/chmod go-w in: " + ", ".join(missing)


def check_devcontainer_workspace_not_world_writable():
    """The devcontainer workspace must not be world-writable."""
    text = _read(".devcontainer/Dockerfile")
    assert not WORLD_WRITABLE_RE.search(text), "devcontainer workspace is world-writable"


def check_readme_non_root_cache_guidance():
    """Each README must document a writable cache directory for non-root runs."""
    missing = []
    for rel_path in READMES:
        text = _read(rel_path)
        if "TORCH_EXTENSIONS_DIR" not in text and "SGLANG_CACHE_DIR" not in text:
            missing.append(rel_path)
    assert not missing, "missing non-root cache guidance in: " + ", ".join(missing)


def check_readme_keyword_hit_threshold():
    """READMEs must collectively mention at least six relevant keywords."""
    hits = 0
    for rel_path in READMES:
        text = _read(rel_path).lower()
        for keyword in README_KEYWORDS:
            if keyword.lower() in text:
                hits += 1
    assert hits >= 6, f"README keyword hits {hits} < 6"


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
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
