#!/usr/bin/env python3
"""Stdlib-only checks for Dockerfile permissions and non-root cache docs.

These checks intentionally avoid importing sglang, pytest, or any third-party
package so they can run under a bare ``python3`` interpreter in CI without
installing the full sglang dependency tree.

Run standalone with::

    python3 test/dockerfile_permissions/run.py

or via pytest with::

    python3 -m pytest -q -c test/dockerfile_permissions/pytest.ini \
        test/dockerfile_permissions/
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Dockerfiles that ship code loaded into a serving/build process and therefore
# must not be world-writable.
DOCKERFILES = [
    "docker/Dockerfile",
    "examples/frontend_language/usage/triton/Dockerfile",
    "python/sglang/kernels/aot/Dockerfile",
    ".devcontainer/Dockerfile",
]

# READMEs that must document how to run as a non-default user with a writable
# cache directory after the permission hardening.
READMES = [
    "README.md",
    "examples/frontend_language/usage/triton/README.md",
    "python/sglang/kernels/aot/README.md",
]

# Matches `chmod 777`, `chmod -R 777`, `chmod -R 0777`, `chmod a+rwx`, etc.
WORLD_WRITABLE_RE = re.compile(
    r"chmod\s+(?:-[A-Za-z]+\s+)*(?:0?777|a\+rwx|ugo\+rwx)\b"
)

# Matches a chown to a non-root user (e.g. `chown -R vllm /workspace`).
CHOWN_RE = re.compile(r"chown\s+(?:-[A-Za-z]+\s+)*[A-Za-z_][\w.-]*")

# Matches `chmod -R go-w` (with optional extra flags).
GO_W_RE = re.compile(r"chmod\s+(?:-[A-Za-z]+\s+)*go-w\b")

# Keywords that indicate non-root cache guidance in a README.
README_KEYWORDS = (
    "cache",
    "chown",
    "chmod",
    "go-w",
    "non-root",
    "nonroot",
    "TORCH_EXTENSIONS_DIR",
    "SGLANG_CACHE_DIR",
    "CCACHE_DIR",
    "PIP_CACHE_DIR",
)

# A fenced code block containing at least one `docker run` invocation.
DOCKER_RUN_RE = re.compile(r"docker\s+run\b")


def _read(rel_path: str) -> str:
    path = REPO_ROOT / rel_path
    if not path.is_file():
        raise AssertionError(f"expected file not found: {rel_path}")
    return path.read_text(encoding="utf-8")


def check_no_world_writable_permissions() -> None:
    """No affected Dockerfile may grant world-writable permissions."""
    offenders = []
    for rel_path in DOCKERFILES:
        text = _read(rel_path)
        for lineno, line in enumerate(text.splitlines(), start=1):
            if WORLD_WRITABLE_RE.search(line):
                offenders.append(f"{rel_path}:{lineno}: {line.strip()}")
    if offenders:
        raise AssertionError(
            "world-writable chmod found in Dockerfiles:\n" + "\n".join(offenders)
        )


def check_minimal_permission_statements() -> None:
    """Affected Dockerfiles must chown a runtime user and chmod go-w."""
    missing = []
    for rel_path in DOCKERFILES:
        text = _read(rel_path)
        if not CHOWN_RE.search(text):
            missing.append(f"{rel_path}: missing chown to a runtime user")
        if not GO_W_RE.search(text):
            missing.append(f"{rel_path}: missing chmod go-w")
    if missing:
        raise AssertionError(
            "minimal-permission statements missing:\n" + "\n".join(missing)
        )


def check_devcontainer_workspace_not_world_writable() -> None:
    """The devcontainer workspace must not be world-writable."""
    text = _read(".devcontainer/Dockerfile")
    for lineno, line in enumerate(text.splitlines(), start=1):
        if WORLD_WRITABLE_RE.search(line):
            raise AssertionError(
                f".devcontainer/Dockerfile:{lineno}: world-writable chmod: {line.strip()}"
            )


def check_readme_non_root_cache_guidance() -> None:
    """Each README must document a writable cache dir for non-root runs."""
    problems = []
    for rel_path in READMES:
        text = _read(rel_path)
        lowered = text.lower()
        if not any(k.lower() in lowered for k in README_KEYWORDS):
            problems.append(f"{rel_path}: no cache/non-root keywords found")
            continue
        if not DOCKER_RUN_RE.search(text):
            problems.append(f"{rel_path}: no `docker run` example found")
    if problems:
        raise AssertionError(
            "README non-root cache guidance missing:\n" + "\n".join(problems)
        )


def check_readme_keyword_hit_threshold() -> None:
    """Across the READMEs, at least 6 cache/permission keyword hits."""
    hits = 0
    for rel_path in READMES:
        text = _read(rel_path)
        lowered = text.lower()
        for keyword in README_KEYWORDS:
            hits += lowered.count(keyword.lower())
    if hits < 6:
        raise AssertionError(
            f"expected >= 6 README keyword hits, found {hits}"
        )


ALL_CHECKS = (
    check_no_world_writable_permissions,
    check_minimal_permission_statements,
    check_devcontainer_workspace_not_world_writable,
    check_readme_non_root_cache_guidance,
    check_readme_keyword_hit_threshold,
)


def run_all() -> int:
    failures = 0
    for check in ALL_CHECKS:
        try:
            check()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {check.__name__}: {exc}")
        else:
            print(f"ok   {check.__name__}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(run_all())
