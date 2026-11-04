#!/usr/bin/env python3
"""Stdlib-only Dockerfile permission and documentation checks.

This module intentionally imports only the Python standard library so CI can
run the checks with a bare ``python3`` interpreter, without installing sglang
or pytest. It is imported by ``test_dockerfile_permissions.py`` (pytest
wrapper) and by ``run.py`` (standalone runner).

Checks:
  * No affected Dockerfile grants world-writable permissions (``chmod 777``
    or ``chmod -R 777``) on ``/workspace`` or the runtime directory.
  * Affected Dockerfiles use the minimal-permission pattern: ``chown`` to the
    runtime user plus ``chmod -R go-w``.
  * The devcontainer workspace is not world-writable.
  * Relevant READMEs document how to run as a non-root user with a writable
    cache directory, including at least one executable command.
"""

import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Dockerfiles that ship code loaded into a serving/build process and therefore
# must not be world-writable.
DOCKERFILES = [
    "docker/Dockerfile",
    "examples/frontend_language/usage/triton/Dockerfile",
    "python/sglang/kernels/aot/Dockerfile",
    ".devcontainer/Dockerfile",
]

# READMEs that must document the non-root writable-cache guidance.
READMES = [
    "README.md",
    "examples/frontend_language/usage/triton/README.md",
    "python/sglang/kernels/aot/README.md",
]

# Matches ``chmod 777``, ``chmod -R 777``, ``chmod -R 0777``, ``chmod a+rwx``
# and similar world-writable grants. We only flag them when they target
# ``/workspace`` or the runtime directory, since unrelated 777 grants (e.g. on
# a throwaway build directory) are out of scope for this check.
WORLD_WRITABLE_RE = re.compile(
    r"chmod\s+(?:-R\s+)?(?:0?777|a\+rwx|ugo\+rwx)\b[^\n]*?(?:/workspace|/sgl-workspace|/opt/sglang|/sgl-kernel)",
    re.IGNORECASE,
)

# Minimal-permission pattern: chown to a runtime user and chmod go-w.
CHOWN_RE = re.compile(r"chown\s+(?:-R\s+)?[^\s]+\s+(?:/workspace|/sgl-workspace|/opt/sglang|/sgl-kernel)")
CHMOD_GO_W_RE = re.compile(r"chmod\s+(?:-R\s+)?go-w\s+(?:/workspace|/sgl-workspace|/opt/sglang|/sgl-kernel)")

# README guidance keywords. We require a minimum number of hits across the
# READMEs so a single incidental mention does not satisfy the check.
README_KEYWORDS = [
    "non-root",
    "--user",
    "TORCH_EXTENSIONS_DIR",
    "SGLANG_CACHE_DIR",
    "chown",
    "chmod",
    "go-w",
    "cache",
]
README_MIN_HITS = 6

# A fenced code block containing a docker run command counts as an executable
# command for the README guidance check.
DOCKER_RUN_RE = re.compile(r"docker\s+run\b")


def _read(rel_path):
    path = os.path.join(REPO_ROOT, rel_path)
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def check_no_world_writable_permissions():
    """No affected Dockerfile may grant world-writable permissions."""
    failures = []
    for rel_path in DOCKERFILES:
        try:
            text = _read(rel_path)
        except FileNotFoundError:
            failures.append(f"{rel_path}: file not found")
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if WORLD_WRITABLE_RE.search(line):
                failures.append(
                    f"{rel_path}:{lineno}: world-writable permission grant: {line.strip()}"
                )
    return failures


def check_minimal_permission_statements():
    """Affected Dockerfiles must use chown + chmod go-w on runtime dirs."""
    failures = []
    for rel_path in DOCKERFILES:
        try:
            text = _read(rel_path)
        except FileNotFoundError:
            failures.append(f"{rel_path}: file not found")
            continue
        if not CHOWN_RE.search(text):
            failures.append(f"{rel_path}: missing chown to runtime user on runtime directory")
        if not CHMOD_GO_W_RE.search(text):
            failures.append(f"{rel_path}: missing chmod -R go-w on runtime directory")
    return failures


def check_devcontainer_workspace_not_world_writable():
    """The devcontainer workspace must not be world-writable."""
    rel_path = ".devcontainer/Dockerfile"
    try:
        text = _read(rel_path)
    except FileNotFoundError:
        return [f"{rel_path}: file not found"]
    failures = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if WORLD_WRITABLE_RE.search(line):
            failures.append(
                f"{rel_path}:{lineno}: world-writable permission grant: {line.strip()}"
            )
    return failures


def check_readme_non_root_cache_guidance():
    """READMEs must document non-root writable-cache guidance with a command."""
    failures = []
    combined = []
    for rel_path in READMES:
        try:
            text = _read(rel_path)
        except FileNotFoundError:
            failures.append(f"{rel_path}: file not found")
            continue
        combined.append(text)
        if not DOCKER_RUN_RE.search(text):
            failures.append(f"{rel_path}: missing executable docker run command")
    joined = "\n".join(combined)
    hits = sum(1 for keyword in README_KEYWORDS if keyword in joined)
    if hits < README_MIN_HITS:
        failures.append(
            f"README guidance keywords: {hits} hits, need >= {README_MIN_HITS}"
        )
    return failures


def check_readme_keyword_hit_threshold():
    """Alias kept for the pytest wrapper's re-export list."""
    return check_readme_non_root_cache_guidance()


CHECKS = [
    check_no_world_writable_permissions,
    check_minimal_permission_statements,
    check_devcontainer_workspace_not_world_writable,
    check_readme_non_root_cache_guidance,
]


def main():
    failures = []
    for check in CHECKS:
        failures.extend(check())
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    print("OK: all Dockerfile permission and README guidance checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
