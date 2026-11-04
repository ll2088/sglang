#!/usr/bin/env python3
"""Stdlib-only checks for Dockerfile permissions and non-root cache docs.

These checks intentionally avoid importing sglang, pytest, or any third-party
package so CI can run them with a bare ``python3`` interpreter. They are used
by both ``run.py`` (standalone) and ``test_dockerfile_permissions.py``
(pytest wrapper).
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
    r"chmod\s+(?:-R\s+)?(?:0?777|a\+rwx|ugo\+rwx)\b", re.IGNORECASE
)

# Matches a chown that targets a non-root user (e.g. `chown -R vllm`, `chown -R devuser:devuser`).
CHOWN_USER_RE = re.compile(r"chown\s+(?:-R\s+)?[A-Za-z_][\w.-]*(?::[\w.-]+)?\s", re.IGNORECASE)

# Matches `chmod -R go-w` (or `chmod go-w`) used to drop group/other write.
CHMOD_GO_W_RE = re.compile(r"chmod\s+(?:-R\s+)?go-w\b", re.IGNORECASE)


def _read(rel_path):
    path = os.path.join(REPO_ROOT, rel_path)
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def _dockerfile_texts():
    return {rel: _read(rel) for rel in DOCKERFILES}


def check_no_world_writable_permissions():
    """No affected Dockerfile may grant world-writable permissions."""
    offenders = []
    for rel, text in _dockerfile_texts().items():
        for lineno, line in enumerate(text.splitlines(), 1):
            if WORLD_WRITABLE_RE.search(line):
                offenders.append(f"{rel}:{lineno}: {line.strip()}")
    assert not offenders, "world-writable chmod found:\n" + "\n".join(offenders)


def check_minimal_permission_statements():
    """Each affected Dockerfile must chown a runtime user and chmod go-w."""
    missing = []
    for rel, text in _dockerfile_texts().items():
        if not CHOWN_USER_RE.search(text):
            missing.append(f"{rel}: missing chown to a runtime user")
        if not CHMOD_GO_W_RE.search(text):
            missing.append(f"{rel}: missing chmod go-w")
    assert not missing, "missing minimal-permission statements:\n" + "\n".join(missing)


def check_devcontainer_workspace_not_world_writable():
    """The devcontainer workspace must not be world-writable."""
    text = _read(".devcontainer/Dockerfile")
    for lineno, line in enumerate(text.splitlines(), 1):
        assert not WORLD_WRITABLE_RE.search(line), (
            f".devcontainer/Dockerfile:{lineno}: world-writable chmod: {line.strip()}"
        )


def check_readme_non_root_cache_guidance():
    """Each README must document a writable cache dir for non-root users."""
    missing = []
    for rel in READMES:
        text = _read(rel)
        lowered = text.lower()
        has_section = (
            "non-root" in lowered
            or "non_default_user" in lowered
            or "非默认用户" in text
        )
        has_cache = "cache" in lowered or "缓存" in text
        has_command = "docker run" in lowered or "TORCH_EXTENSIONS_DIR" in text
        if not (has_section and has_cache and has_command):
            missing.append(
                f"{rel}: needs non-root section, cache guidance, and an executable command"
            )
    assert not missing, "missing README guidance:\n" + "\n".join(missing)


def check_readme_keyword_hit_threshold(min_hits=6):
    """The combined READMEs must mention the relevant keywords often enough."""
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
    assert hits >= min_hits, f"README keyword hits {hits} < {min_hits}"


ALL_CHECKS = [
    check_no_world_writable_permissions,
    check_minimal_permission_statements,
    check_devcontainer_workspace_not_world_writable,
    check_readme_non_root_cache_guidance,
    check_readme_keyword_hit_threshold,
]
