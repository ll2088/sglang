#!/usr/bin/env python3
"""Stdlib-only checks for Dockerfile permission hardening and README guidance.

This module must not import sglang or any third-party package so it can run
under a bare ``python3`` interpreter in CI without installing the project
dependencies. It is imported by both ``run.py`` (standalone) and
``test_dockerfile_permissions.py`` (pytest wrapper).
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
    r"chmod\s+(?:-R\s+)?(?:0?777|a\+rwx|ugo\+rwx|o\+w)\b"
)

# Matches a chown that targets a non-root user (e.g. `chown -R vllm /workspace`).
CHOWN_RE = re.compile(r"chown\s+(?:-R\s+)?(?!root\b)[A-Za-z0-9_.-]+(?::[A-Za-z0-9_.-]+)?\s+/")

# Matches `chmod -R go-w` (or `chmod go-w`) hardening statements.
GO_W_RE = re.compile(r"chmod\s+(?:-R\s+)?go-w\b")


def _read(rel_path):
    with open(os.path.join(REPO_ROOT, rel_path), "r", encoding="utf-8") as f:
        return f.read()


def check_no_world_writable_permissions():
    """No Dockerfile may grant world-writable permissions."""
    offenders = []
    for rel in DOCKERFILES:
        text = _read(rel)
        for lineno, line in enumerate(text.splitlines(), 1):
            if WORLD_WRITABLE_RE.search(line):
                offenders.append(f"{rel}:{lineno}: {line.strip()}")
    assert not offenders, "world-writable chmod found:\n" + "\n".join(offenders)


def check_minimal_permission_statements():
    """Each Dockerfile must chown to a runtime user and apply chmod go-w."""
    missing = []
    for rel in DOCKERFILES:
        text = _read(rel)
        if not CHOWN_RE.search(text):
            missing.append(f"{rel}: missing chown to runtime user")
        if not GO_W_RE.search(text):
            missing.append(f"{rel}: missing chmod go-w")
    assert not missing, "missing minimal-permission statements:\n" + "\n".join(missing)


def check_devcontainer_workspace_not_world_writable():
    """.devcontainer must not make /sgl-workspace world-writable."""
    text = _read(".devcontainer/Dockerfile")
    for lineno, line in enumerate(text.splitlines(), 1):
        if "/sgl-workspace" in line and WORLD_WRITABLE_RE.search(line):
            raise AssertionError(
                f".devcontainer/Dockerfile:{lineno}: world-writable workspace: {line.strip()}"
            )


def check_readme_non_root_cache_guidance():
    """Each README must document a writable cache dir for non-root users."""
    missing = []
    for rel in READMES:
        text = _read(rel)
        lowered = text.lower()
        has_section = (
            "non-root" in lowered
            or "non-default user" in lowered
            or "非默认用户" in text
        )
        has_cache = "cache" in lowered or "缓存" in text
        has_command = "docker run" in text or "TORCH_EXTENSIONS_DIR" in text
        if not (has_section and has_cache and has_command):
            missing.append(
                f"{rel}: needs non-root cache guidance with an executable command"
            )
    assert not missing, "missing README guidance:\n" + "\n".join(missing)


def check_readme_keyword_hit_threshold():
    """READMEs must mention at least 6 relevant keywords in total."""
    keywords = [
        "cache",
        "缓存",
        "chown",
        "chmod",
        "go-w",
        "非默认用户",
        "non-root",
    ]
    hits = 0
    for rel in READMES:
        text = _read(rel)
        lowered = text.lower()
        for kw in keywords:
            if kw.lower() in lowered:
                hits += 1
    assert hits >= 6, f"README keyword hits {hits} < 6"


ALL_CHECKS = [
    check_no_world_writable_permissions,
    check_minimal_permission_statements,
    check_devcontainer_workspace_not_world_writable,
    check_readme_non_root_cache_guidance,
    check_readme_keyword_hit_threshold,
]
