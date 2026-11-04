#!/usr/bin/env python3
"""Stdlib-only checks for Dockerfile permission hardening and README guidance.

These checks intentionally import nothing outside the Python standard library so
CI can run them with a bare ``python3`` interpreter, without installing sglang or
any of its dependencies. They are also re-exported as ``test_*`` functions by
``test_dockerfile_permissions.py`` for pytest collection.

Run standalone:
    python3 test/dockerfile_permissions/checks.py
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

# Matches `chmod 777`, `chmod -R 777`, `chmod -R 0777`, `chmod a+rwx`, etc.
WORLD_WRITABLE_RE = re.compile(
    r"chmod\s+(?:-R\s+)?(?:0?777|a\+rwx|ugo\+rwx|o\+w)\b",
    re.IGNORECASE,
)

# Matches a chown to a non-root user (e.g. `chown -R vllm /workspace`).
CHOWN_USER_RE = re.compile(r"chown\s+(?:-R\s+)?[A-Za-z_][\w.-]*\s", re.IGNORECASE)

# Matches `chmod -R go-w` (or `chmod go-w`) on a path.
CHMOD_GO_W_RE = re.compile(r"chmod\s+(?:-R\s+)?go-w\b", re.IGNORECASE)


def _read(rel_path):
    path = os.path.join(REPO_ROOT, rel_path)
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        return fh.read()


def _iter_dockerfiles():
    for rel in DOCKERFILES:
        path = os.path.join(REPO_ROOT, rel)
        if os.path.isfile(path):
            yield rel, _read(rel)


def check_no_world_writable_permissions():
    """No affected Dockerfile may grant world-writable permissions."""
    offenders = []
    for rel, text in _iter_dockerfiles():
        for lineno, line in enumerate(text.splitlines(), start=1):
            if WORLD_WRITABLE_RE.search(line):
                offenders.append(f"{rel}:{lineno}: {line.strip()}")
    assert not offenders, "world-writable chmod found:\n" + "\n".join(offenders)


def check_minimal_permission_statements():
    """Affected Dockerfiles must chown a runtime user and chmod go-w."""
    missing = []
    for rel, text in _iter_dockerfiles():
        if not CHOWN_USER_RE.search(text):
            missing.append(f"{rel}: missing chown to a runtime user")
        if not CHMOD_GO_W_RE.search(text):
            missing.append(f"{rel}: missing chmod go-w")
    assert not missing, "missing minimal-permission statements:\n" + "\n".join(missing)


def check_devcontainer_workspace_not_world_writable():
    """The devcontainer must not make /sgl-workspace world-writable."""
    rel = ".devcontainer/Dockerfile"
    path = os.path.join(REPO_ROOT, rel)
    if not os.path.isfile(path):
        return
    text = _read(rel)
    for lineno, line in enumerate(text.splitlines(), start=1):
        if "/sgl-workspace" in line and WORLD_WRITABLE_RE.search(line):
            raise AssertionError(f"{rel}:{lineno}: {line.strip()}")


def check_readme_non_root_cache_guidance():
    """Each README must document a writable cache dir for non-root runs."""
    missing = []
    for rel in READMES:
        path = os.path.join(REPO_ROOT, rel)
        if not os.path.isfile(path):
            continue
        text = _read(rel)
        lowered = text.lower()
        has_section = (
            "non-root" in lowered
            or "non default user" in lowered
            or "non-default user" in lowered
        )
        has_cache = "cache" in lowered or "缓存" in text
        has_command = "docker run" in lowered
        if not (has_section and has_cache and has_command):
            missing.append(
                f"{rel}: section={has_section} cache={has_cache} command={has_command}"
            )
    assert not missing, "README guidance missing:\n" + "\n".join(missing)


def check_readme_keyword_hit_threshold():
    """READMEs must mention at least 6 relevant permission/cache keywords."""
    keywords = [
        "cache",
        "缓存",
        "chown",
        "chmod",
        "go-w",
        "non-root",
        "非默认用户",
    ]
    hits = 0
    for rel in READMES:
        path = os.path.join(REPO_ROOT, rel)
        if not os.path.isfile(path):
            continue
        text = _read(rel)
        lowered = text.lower()
        for kw in keywords:
            if kw.lower() in lowered or kw in text:
                hits += 1
    assert hits >= 6, f"only {hits} README keyword hits, need >= 6"


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
