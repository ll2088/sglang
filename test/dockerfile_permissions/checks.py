#!/usr/bin/env python3
"""Stdlib-only checks for Dockerfile permissions and non-root cache docs.

These checks intentionally avoid importing sglang or pytest at module scope so
CI can run them with a bare ``python3`` interpreter. They are also re-exported
by ``test_dockerfile_permissions.py`` for pytest collection.

Run standalone with::

    python3 test/dockerfile_permissions/run.py
"""

import os
import re
import sys


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

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

# Matches a chown to a non-root user (e.g. `chown -R vllm /workspace`).
CHOWN_RE = re.compile(r"chown\s+(?:-R\s+)?[A-Za-z0-9_.-]+\s+", re.IGNORECASE)

# Matches `chmod -R go-w` (with optional extra flags).
GO_W_RE = re.compile(r"chmod\s+(?:-R\s+)?go-w\b", re.IGNORECASE)

# Keywords that indicate non-root cache guidance in a README.
README_KEYWORDS = [
    "cache",
    "chown",
    "chmod",
    "go-w",
    "non-root",
    "TORCH_EXTENSIONS_DIR",
    "SGLANG_CACHE_DIR",
]

# A fenced code block containing a docker run command.
DOCKER_RUN_RE = re.compile(r"docker\s+run\b", re.IGNORECASE)


def _read(path):
    full = os.path.join(REPO_ROOT, path)
    with open(full, "r", encoding="utf-8") as fh:
        return fh.read()


def _iter_dockerfiles():
    for rel in DOCKERFILES:
        full = os.path.join(REPO_ROOT, rel)
        if not os.path.isfile(full):
            raise AssertionError("expected Dockerfile not found: %s" % rel)
        yield rel, _read(rel)


def check_no_world_writable_permissions():
    """No Dockerfile may grant world-writable permissions."""
    offenders = []
    for rel, text in _iter_dockerfiles():
        for lineno, line in enumerate(text.splitlines(), 1):
            if WORLD_WRITABLE_RE.search(line):
                offenders.append("%s:%d: %s" % (rel, lineno, line.strip()))
    assert not offenders, "world-writable chmod found:\n" + "\n".join(offenders)


def check_minimal_permission_statements():
    """Each Dockerfile must chown a runtime user and drop group/other write."""
    missing = []
    for rel, text in _iter_dockerfiles():
        if not CHOWN_RE.search(text):
            missing.append("%s: missing chown to runtime user" % rel)
        if not GO_W_RE.search(text):
            missing.append("%s: missing chmod go-w" % rel)
    assert not missing, "minimal permission statements missing:\n" + "\n".join(missing)


def check_devcontainer_workspace_not_world_writable():
    """The devcontainer workspace must not be world-writable."""
    rel = ".devcontainer/Dockerfile"
    text = _read(rel)
    for lineno, line in enumerate(text.splitlines(), 1):
        if WORLD_WRITABLE_RE.search(line):
            raise AssertionError(
                "%s:%d grants world-writable permissions: %s"
                % (rel, lineno, line.strip())
            )


def check_readme_non_root_cache_guidance():
    """Each README must document a writable cache dir for non-root users."""
    problems = []
    for rel in READMES:
        full = os.path.join(REPO_ROOT, rel)
        if not os.path.isfile(full):
            problems.append("%s: missing README" % rel)
            continue
        text = _read(rel)
        lowered = text.lower()
        if "non-root" not in lowered and "non-root user" not in lowered:
            problems.append("%s: missing non-root section" % rel)
        if not DOCKER_RUN_RE.search(text):
            problems.append("%s: missing docker run example" % rel)
        if "cache" not in lowered:
            problems.append("%s: missing cache guidance" % rel)
    assert not problems, "README guidance missing:\n" + "\n".join(problems)


def check_readme_keyword_hit_threshold(min_hits=6):
    """READMEs must collectively mention enough cache/permission keywords."""
    hits = 0
    for rel in READMES:
        full = os.path.join(REPO_ROOT, rel)
        if not os.path.isfile(full):
            continue
        text = _read(rel)
        lowered = text.lower()
        for kw in README_KEYWORDS:
            if kw.lower() in lowered:
                hits += 1
    assert hits >= min_hits, (
        "README keyword hits %d < required %d" % (hits, min_hits)
    )


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
            sys.stderr.write("FAIL %s: %s\n" % (check.__name__, exc))
        else:
            sys.stdout.write("PASS %s\n" % check.__name__)
    if failures:
        sys.stderr.write("%d check(s) failed\n" % failures)
        return 1
    sys.stdout.write("all checks passed\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
