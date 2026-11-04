#!/usr/bin/env python3
"""Stdlib-only checks for Dockerfile permission hardening and README guidance.

These checks intentionally import nothing outside the Python standard library so
CI can run them with a bare ``python3`` interpreter, without installing sglang or
any third-party package. They are shared by:

* ``test/test_dockerfile_permissions.py`` (standalone runner + pytest entry)
* ``test/dockerfile_permissions/run.py`` (standalone runner)
* ``test/dockerfile_permissions/test_dockerfile_permissions.py`` (pytest wrapper)

Do not add third-party imports here; doing so reintroduces the full sglang
dependency install that these checks are designed to avoid.
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
    r"chmod\s+(?:-[A-Za-z]+\s+)*(?:0?777|a\+rwx|ugo\+rwx)\b"
)

# Matches a chown that targets a non-root runtime user (e.g. `chown -R vllm`,
# `chown -R devuser:devuser`, `chown -R root:root`).
CHOWN_RE = re.compile(r"chown\s+(?:-[A-Za-z]+\s+)*[A-Za-z0-9_.-]+(?::[A-Za-z0-9_.-]+)?\s+/")

# Matches `chmod -R go-w` (optionally with other flags) on an absolute path.
GO_W_RE = re.compile(r"chmod\s+(?:-[A-Za-z]+\s+)*go-w\s+/")

README_KEYWORDS = [
    "cache",
    "chown",
    "chmod",
    "go-w",
    "non-root",
    "TORCH_EXTENSIONS_DIR",
    "SGLANG_CACHE_DIR",
    "CCACHE_DIR",
    "PIP_CACHE_DIR",
]


def _read(rel_path):
    with open(os.path.join(REPO_ROOT, rel_path), "r", encoding="utf-8") as f:
        return f.read()


def check_no_world_writable_permissions():
    """No affected Dockerfile may grant world-writable permissions."""
    offenders = []
    for rel_path in DOCKERFILES:
        text = _read(rel_path)
        for lineno, line in enumerate(text.splitlines(), start=1):
            if WORLD_WRITABLE_RE.search(line):
                offenders.append(f"{rel_path}:{lineno}: {line.strip()}")
    assert not offenders, (
        "world-writable chmod found in Dockerfiles; use chown <runtime-user> "
        "+ chmod go-w instead:\n" + "\n".join(offenders)
    )


def check_minimal_permission_statements():
    """Each affected Dockerfile must chown a runtime user and chmod go-w."""
    missing = []
    for rel_path in DOCKERFILES:
        text = _read(rel_path)
        if not CHOWN_RE.search(text):
            missing.append(f"{rel_path}: missing chown <runtime-user> ...")
        if not GO_W_RE.search(text):
            missing.append(f"{rel_path}: missing chmod ... go-w ...")
    assert not missing, (
        "Dockerfiles must use minimal permissions (chown runtime user + chmod go-w):\n"
        + "\n".join(missing)
    )


def check_devcontainer_workspace_not_world_writable():
    """The devcontainer workspace must not be world-writable."""
    text = _read(".devcontainer/Dockerfile")
    for lineno, line in enumerate(text.splitlines(), start=1):
        if WORLD_WRITABLE_RE.search(line):
            raise AssertionError(
                f".devcontainer/Dockerfile:{lineno}: world-writable chmod: {line.strip()}"
            )


def check_readme_non_root_cache_guidance():
    """Each README must document a writable cache dir for non-root runs."""
    missing = []
    for rel_path in READMES:
        text = _read(rel_path)
        lowered = text.lower()
        if "non-root" not in lowered and "non-root user" not in lowered:
            missing.append(f"{rel_path}: missing non-root user guidance")
        if "cache" not in lowered:
            missing.append(f"{rel_path}: missing cache directory guidance")
        if "docker run" not in text:
            missing.append(f"{rel_path}: missing executable docker run example")
    assert not missing, (
        "READMEs must document a writable cache directory for non-root runs:\n"
        + "\n".join(missing)
    )


def check_readme_keyword_hit_threshold(min_hits=6):
    """The READMEs must collectively mention enough permission/cache keywords."""
    hits = 0
    for rel_path in READMES:
        text = _read(rel_path)
        for keyword in README_KEYWORDS:
            hits += text.count(keyword)
    assert hits >= min_hits, (
        f"README permission/cache keyword hits = {hits}, expected >= {min_hits}"
    )


ALL_CHECKS = [
    check_no_world_writable_permissions,
    check_minimal_permission_statements,
    check_devcontainer_workspace_not_world_writable,
    check_readme_non_root_cache_guidance,
    check_readme_keyword_hit_threshold,
]


def run_all():
    for check in ALL_CHECKS:
        check()


if __name__ == "__main__":
    run_all()
    print("dockerfile permission checks passed")
