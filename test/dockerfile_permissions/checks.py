#!/usr/bin/env python3
"""Stdlib-only checks for Dockerfile permission hardening and README guidance.

These checks intentionally avoid importing sglang or any third-party package so
CI can run them with a bare python3 interpreter (or with pytest only). They
assert that the affected Dockerfiles do not grant world-writable permissions on
runtime directories and instead use a minimal-permission pattern, and that the
related READMEs document a writable cache directory for non-default users.
"""

import os
import re

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

# Matches `chmod 777`, `chmod -R 777`, `chmod -R 0777`, `chmod a+rwx`, etc.
WORLD_WRITABLE_RE = re.compile(
    r"chmod\s+(?:-R\s+)?(?:0?777|a\+rwx|ugo\+rwx|\+rwx)\b"
)

# Matches a chown to a non-root runtime user (e.g. `chown -R vllm /workspace`).
CHOWN_RE = re.compile(r"chown\s+(?:-R\s+)?[A-Za-z0-9_.-]+(?::[A-Za-z0-9_.-]+)?\s+/")

# Matches `chmod -R go-w` (or `chmod go-w`) on a path.
GO_W_RE = re.compile(r"chmod\s+(?:-R\s+)?go-w\b")


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
        if not CHOWN_RE.search(text):
            missing.append(f"{rel_path}: missing chown to runtime user")
        if not GO_W_RE.search(text):
            missing.append(f"{rel_path}: missing chmod go-w")
    assert not missing, "minimal-permission statements missing:\n" + "\n".join(missing)


def check_devcontainer_workspace_not_world_writable():
    """The devcontainer workspace must not be world-writable."""
    text = _read(".devcontainer/Dockerfile")
    assert not WORLD_WRITABLE_RE.search(text), (
        ".devcontainer/Dockerfile grants world-writable permissions"
    )
    assert GO_W_RE.search(text), ".devcontainer/Dockerfile missing chmod go-w"


def check_readme_non_root_cache_guidance():
    """READMEs must document a writable cache directory for non-root users."""
    missing = []
    for rel_path in READMES:
        text = _read(rel_path)
        lowered = text.lower()
        has_section = (
            "non-root" in lowered
            or "non-default" in lowered
            or "非默认用户" in text
        )
        has_cache = "cache" in lowered or "缓存" in text
        has_command = "docker run" in lowered or "mkdir -p" in lowered
        if not (has_section and has_cache and has_command):
            missing.append(
                f"{rel_path}: needs non-root section, cache guidance and a runnable command"
            )
    assert not missing, "README guidance missing:\n" + "\n".join(missing)


def check_readme_keyword_hit_threshold(min_hits=6):
    """The READMEs must contain enough cache/permission keywords."""
    keywords = (
        "cache",
        "缓存",
        "chown",
        "chmod",
        "go-w",
        "非默认用户",
        "non-root",
    )
    hits = 0
    for rel_path in READMES:
        text = _read(rel_path)
        lowered = text.lower()
        for keyword in keywords:
            needle = keyword.lower()
            hits += lowered.count(needle)
    assert hits >= min_hits, f"README keyword hits {hits} < {min_hits}"
