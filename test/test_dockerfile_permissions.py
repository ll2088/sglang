#!/usr/bin/env python3
"""Stdlib-only checks for Dockerfile permission hardening and README guidance.

This module intentionally imports nothing outside the Python standard library so
it can run under a bare ``python3`` interpreter without installing sglang or
pytest. It verifies that:

* the affected Dockerfiles no longer grant world-writable permissions on the
  runtime workspace (``chmod -R 777`` / ``chmod 777``), and
* they instead use a minimal-permission pattern (``chown`` to the runtime user
  plus ``chmod -R go-w``), and
* the related READMEs document how to run as a non-default user with a writable
  cache directory, including at least one executable command.

Usage:
    python3 test/test_dockerfile_permissions.py
"""

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DOCKERFILES = [
    REPO_ROOT / "docker" / "Dockerfile",
    REPO_ROOT / "examples" / "frontend_language" / "usage" / "triton" / "Dockerfile",
    REPO_ROOT / "python" / "sglang" / "kernels" / "aot" / "Dockerfile",
    REPO_ROOT / ".devcontainer" / "Dockerfile",
]

READMES = [
    REPO_ROOT / "README.md",
    REPO_ROOT / "examples" / "frontend_language" / "usage" / "triton" / "README.md",
    REPO_ROOT / "python" / "sglang" / "kernels" / "aot" / "README.md",
]

# Matches `chmod 777`, `chmod -R 777`, `chmod -R 0777`, `chmod -R a+rwx`, etc.
_WORLD_WRITABLE_RE = re.compile(
    r"chmod\s+(?:-R\s+)?(?:0?777|a\+rwx|ugo\+rwx)\b",
    re.IGNORECASE,
)

# Matches a chown to a non-root runtime user (e.g. `chown -R vllm /workspace`).
_CHOWN_RE = re.compile(r"chown\s+(?:-R\s+)?[A-Za-z0-9_.-]+(?::[A-Za-z0-9_.-]+)?\s+/", re.IGNORECASE)

# Matches `chmod -R go-w` (or `chmod -R g-w,o-w`).
_GO_W_RE = re.compile(r"chmod\s+-R\s+(?:go-w|g-w,o-w|o-w,g-w)\b", re.IGNORECASE)

# README guidance keywords.
_README_KEYWORDS = (
    "cache",
    "chown",
    "chmod",
    "go-w",
    "non-root",
    "non-default",
    "TORCH_EXTENSIONS_DIR",
    "SGLANG_CACHE_DIR",
    "CCACHE_DIR",
    "PIP_CACHE_DIR",
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def missing_paths() -> list:
    return [str(p) for p in (*DOCKERFILES, *READMES) if not p.is_file()]


def test_no_world_writable_permissions() -> None:
    offenders = []
    for path in DOCKERFILES:
        if not path.is_file():
            continue
        for lineno, line in enumerate(_read(path).splitlines(), start=1):
            if _WORLD_WRITABLE_RE.search(line):
                offenders.append(f"{path}:{lineno}: {line.strip()}")
    assert not offenders, "world-writable chmod found:\n" + "\n".join(offenders)


def test_minimal_permission_statements() -> None:
    missing = []
    for path in DOCKERFILES:
        if not path.is_file():
            missing.append(f"{path}: file missing")
            continue
        text = _read(path)
        if not _CHOWN_RE.search(text):
            missing.append(f"{path}: no chown to runtime user")
        if not _GO_W_RE.search(text):
            missing.append(f"{path}: no chmod -R go-w")
    assert not missing, "missing minimal-permission statements:\n" + "\n".join(missing)


def readme_keyword_hits_per_file() -> dict:
    hits = {}
    for path in READMES:
        if not path.is_file():
            hits[str(path)] = 0
            continue
        text = _read(path).lower()
        hits[str(path)] = sum(text.count(kw.lower()) for kw in _README_KEYWORDS)
    return hits


def readme_keyword_hit_count() -> int:
    return sum(readme_keyword_hits_per_file().values())


def test_readme_non_root_cache_guidance() -> None:
    missing = []
    for path in READMES:
        if not path.is_file():
            missing.append(f"{path}: file missing")
            continue
        text = _read(path)
        lowered = text.lower()
        has_guidance = any(kw.lower() in lowered for kw in _README_KEYWORDS)
        has_command = "docker run" in lowered or "mkdir -p" in lowered
        if not has_guidance:
            missing.append(f"{path}: no non-root cache guidance keywords")
        if not has_command:
            missing.append(f"{path}: no executable command")
    assert not missing, "README guidance missing:\n" + "\n".join(missing)


def test_devcontainer_workspace_not_world_writable() -> None:
    path = REPO_ROOT / ".devcontainer" / "Dockerfile"
    if not path.is_file():
        return
    text = _read(path)
    assert not _WORLD_WRITABLE_RE.search(text), f"{path}: world-writable chmod found"
    assert _GO_W_RE.search(text), f"{path}: no chmod -R go-w"


CHECKS = (
    test_no_world_writable_permissions,
    test_minimal_permission_statements,
    test_readme_non_root_cache_guidance,
    test_devcontainer_workspace_not_world_writable,
)


def _run_standalone() -> int:
    missing = missing_paths()
    if missing:
        print("missing paths:")
        for item in missing:
            print(f"  {item}")
        return 1

    failures = 0
    for check in CHECKS:
        try:
            check()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {check.__name__}: {exc}")
        else:
            print(f"PASS {check.__name__}")

    print(f"readme_keyword_hits_total={readme_keyword_hit_count()}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(_run_standalone())
