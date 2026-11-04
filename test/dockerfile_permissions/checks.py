#!/usr/bin/env python3
"""Stdlib-only checks for Dockerfile permission hardening and README guidance.

This module intentionally imports nothing outside the Python standard library so
CI can run it with a bare ``python3`` interpreter, without installing sglang or
pytest. The canonical entry points are::

    python3 test/dockerfile_permissions/checks.py
    python3 test/dockerfile_permissions/run.py
    python3 -m pytest -q -c test/dockerfile_permissions/pytest.ini \
        test/dockerfile_permissions/

Each check returns a list of human-readable failure strings; an empty list means
the check passed. ``main()`` aggregates all checks and returns a process exit
code so the module can be used directly from CI.
"""

import os
import re
import sys

# Repository root is two levels up from this file (test/dockerfile_permissions/).
_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))

# Dockerfiles that ship code loaded into a serving/build process and therefore
# must not be world-writable.
DOCKERFILES = [
    "docker/Dockerfile",
    "examples/frontend_language/usage/triton/Dockerfile",
    "python/sglang/kernels/aot/Dockerfile",
    ".devcontainer/Dockerfile",
]

# READMEs that document how to run the image as a non-default user.
READMES = [
    "README.md",
    "examples/frontend_language/usage/triton/README.md",
    "python/sglang/kernels/aot/README.md",
]

# Matches `chmod 777`, `chmod -R 777`, `chmod -R 0777`, `chmod a+rwx`, etc.
_WORLD_WRITABLE_RE = re.compile(
    r"chmod\s+(?:-R\s+)?(?:0?777|a\+rwx|ugo\+rwx)\b", re.IGNORECASE
)

# Matches a chown that targets a non-root user (e.g. `chown -R vllm /workspace`).
_CHOWN_RE = re.compile(r"chown\s+(?:-R\s+)?[A-Za-z0-9_.-]+(?::[A-Za-z0-9_.-]+)?\s+", re.IGNORECASE)

# Matches `chmod -R go-w` (or `chmod go-w`) on a path.
_GO_W_RE = re.compile(r"chmod\s+(?:-R\s+)?go-w\b", re.IGNORECASE)

# Keywords that indicate non-default-user cache guidance in a README.
_README_KEYWORDS = [
    "cache",
    "chown",
    "chmod",
    "go-w",
    "non-root",
    "TORCH_EXTENSIONS_DIR",
    "SGLANG_CACHE_DIR",
    "--user",
]


def _read(path):
    full = os.path.join(REPO_ROOT, path)
    with open(full, "r", encoding="utf-8") as handle:
        return handle.read()


def check_no_world_writable_permissions():
    """No affected Dockerfile may grant world-writable permissions."""
    failures = []
    for path in DOCKERFILES:
        try:
            text = _read(path)
        except FileNotFoundError:
            failures.append(f"{path}: file not found")
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if _WORLD_WRITABLE_RE.search(line):
                failures.append(
                    f"{path}:{lineno}: world-writable chmod is not allowed: {line.strip()}"
                )
    return failures


def check_minimal_permission_statements():
    """Each affected Dockerfile must chown a runtime user and apply chmod go-w."""
    failures = []
    for path in DOCKERFILES:
        try:
            text = _read(path)
        except FileNotFoundError:
            failures.append(f"{path}: file not found")
            continue
        if not _CHOWN_RE.search(text):
            failures.append(f"{path}: missing chown to a runtime user")
        if not _GO_W_RE.search(text):
            failures.append(f"{path}: missing chmod go-w")
    return failures


def check_devcontainer_workspace_not_world_writable():
    """The devcontainer workspace must not be world-writable."""
    path = ".devcontainer/Dockerfile"
    try:
        text = _read(path)
    except FileNotFoundError:
        return [f"{path}: file not found"]
    failures = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if _WORLD_WRITABLE_RE.search(line):
            failures.append(
                f"{path}:{lineno}: world-writable chmod is not allowed: {line.strip()}"
            )
    return failures


def check_readme_non_root_cache_guidance():
    """Each README must document a writable cache directory for non-root users."""
    failures = []
    for path in READMES:
        try:
            text = _read(path)
        except FileNotFoundError:
            failures.append(f"{path}: file not found")
            continue
        lowered = text.lower()
        if "non-root" not in lowered and "non-default" not in lowered:
            failures.append(f"{path}: missing non-root/non-default user section")
        if "cache" not in lowered:
            failures.append(f"{path}: missing cache directory guidance")
        if "docker run" not in lowered:
            failures.append(f"{path}: missing an executable docker run example")
    return failures


def check_readme_keyword_hit_threshold(min_hits=6):
    """The READMEs together must mention at least ``min_hits`` guidance keywords."""
    hits = 0
    for path in READMES:
        try:
            text = _read(path)
        except FileNotFoundError:
            continue
        lowered = text.lower()
        for keyword in _README_KEYWORDS:
            if keyword.lower() in lowered:
                hits += 1
    if hits < min_hits:
        return [f"README guidance keywords: found {hits}, need at least {min_hits}"]
    return []


CHECKS = [
    check_no_world_writable_permissions,
    check_minimal_permission_statements,
    check_devcontainer_workspace_not_world_writable,
    check_readme_non_root_cache_guidance,
    check_readme_keyword_hit_threshold,
]


def main():
    failures = []
    for check in CHECKS:
        failures.extend(check())
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    print("All Dockerfile permission and README guidance checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
