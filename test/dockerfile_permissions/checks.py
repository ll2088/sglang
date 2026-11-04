#!/usr/bin/env python3
"""Stdlib-only checks for Dockerfile permissions and non-root cache docs.

These checks intentionally avoid importing sglang or pytest so CI can run them
with a bare ``python3`` interpreter (no ``pip install -e python``). Entry points:

    python3 test/dockerfile_permissions/run.py
    python3 test/test_dockerfile_permissions.py
    python3 -m pytest -q -c test/dockerfile_permissions/pytest.ini \
        test/dockerfile_permissions/

Each check returns a list of human-readable failure messages; an empty list
means the check passed. ``main()`` prints every failure and exits non-zero if
any check failed.
"""

import os
import re
import sys

# Repository root: this file lives at <root>/test/dockerfile_permissions/checks.py
_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))

# Dockerfiles that ship code loaded into a serving/build process.
DOCKERFILES = [
    "docker/Dockerfile",
    "examples/frontend_language/usage/triton/Dockerfile",
    "python/sglang/kernels/aot/Dockerfile",
    ".devcontainer/Dockerfile",
]

# READMEs that must document running as a non-default user.
READMES = [
    "README.md",
    "examples/frontend_language/usage/triton/README.md",
    "python/sglang/kernels/aot/README.md",
]

# Matches `chmod 777`, `chmod -R 777`, `chmod -R 0777`, `chmod a+rwx`, etc.
_WORLD_WRITABLE_RE = re.compile(
    r"chmod\s+(?:-[A-Za-z]+\s+)*(?:0?777|a\+rwx|ugo\+rwx)\b"
)

# Matches a minimal-permission statement: chown to a user plus chmod go-w.
_CHOWN_RE = re.compile(r"chown\s+(?:-[A-Za-z]+\s+)*[^\s;&|]+")
_GO_W_RE = re.compile(r"chmod\s+(?:-[A-Za-z]+\s+)*go-w\b")

# Keywords that indicate non-default-user cache guidance in a README.
_README_KEYWORDS = (
    "cache",
    "chown",
    "chmod",
    "go-w",
    "non-root",
    "TORCH_EXTENSIONS_DIR",
    "SGLANG_CACHE_DIR",
    "--user",
)


def _read(rel_path):
    path = os.path.join(REPO_ROOT, rel_path)
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _strip_comments(text):
    """Drop Dockerfile comment lines so commented-out commands are ignored."""
    return "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )


def check_no_world_writable_permissions():
    """No affected Dockerfile may grant world-writable permissions."""
    failures = []
    for rel_path in DOCKERFILES:
        text = _strip_comments(_read(rel_path))
        for match in _WORLD_WRITABLE_RE.finditer(text):
            line_no = text[: match.start()].count("\n") + 1
            failures.append(
                "%s:%d: world-writable permission %r"
                % (rel_path, line_no, match.group(0))
            )
    return failures


def check_minimal_permission_statements():
    """Each affected Dockerfile must chown a user and chmod go-w."""
    failures = []
    for rel_path in DOCKERFILES:
        text = _strip_comments(_read(rel_path))
        if not _CHOWN_RE.search(text):
            failures.append("%s: missing chown to a runtime user" % rel_path)
        if not _GO_W_RE.search(text):
            failures.append("%s: missing chmod go-w" % rel_path)
    return failures


def check_devcontainer_workspace_not_world_writable():
    """The devcontainer workspace must not be world-writable."""
    text = _strip_comments(_read(".devcontainer/Dockerfile"))
    failures = []
    if _WORLD_WRITABLE_RE.search(text):
        failures.append(".devcontainer/Dockerfile: workspace is world-writable")
    if "chmod -R go-w /sgl-workspace" not in text:
        failures.append(
            ".devcontainer/Dockerfile: missing 'chmod -R go-w /sgl-workspace'"
        )
    return failures


def check_readme_non_root_cache_guidance():
    """Each README must document a writable cache dir for non-root runs."""
    failures = []
    for rel_path in READMES:
        text = _read(rel_path)
        lowered = text.lower()
        if "non-root" not in lowered and "non-default user" not in lowered:
            failures.append("%s: missing non-root/non-default-user section" % rel_path)
        if "docker run" not in text:
            failures.append("%s: missing an executable docker run command" % rel_path)
        if "cache" not in lowered:
            failures.append("%s: missing cache directory guidance" % rel_path)
    return failures


def check_readme_keyword_hit_threshold(min_hits=6):
    """READMEs must collectively mention enough cache/permission keywords."""
    combined = "\n".join(_read(rel_path) for rel_path in READMES).lower()
    hits = sum(1 for keyword in _README_KEYWORDS if keyword.lower() in combined)
    if hits < min_hits:
        return [
            "README keyword hits %d < required %d" % (hits, min_hits)
        ]
    return []


CHECKS = (
    check_no_world_writable_permissions,
    check_minimal_permission_statements,
    check_devcontainer_workspace_not_world_writable,
    check_readme_non_root_cache_guidance,
    check_readme_keyword_hit_threshold,
)


def main():
    failures = []
    for check in CHECKS:
        failures.extend(check())
    if failures:
        for failure in failures:
            print("FAIL: %s" % failure)
        return 1
    print("OK: %d Dockerfile permission checks passed" % len(CHECKS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
