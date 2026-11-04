import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

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

WORLD_WRITABLE = re.compile(r"chmod\s+(-R\s+)?777")
WORKSPACE_WORLD_WRITABLE = re.compile(
    r"chmod\s+(-R\s+)?(a\+w|ugo\+w|o\+w|777)\s+[^\n]*/workspace"
)
MIN_PERMISSION = re.compile(r"chmod\s+-R\s+go-w")
CHOWN = re.compile(r"chown\s+-R\s+\S+")

HARDENED_DOCKERFILES = [
    REPO_ROOT / "docker" / "Dockerfile",
    REPO_ROOT / "examples" / "frontend_language" / "usage" / "triton" / "Dockerfile",
    REPO_ROOT / "python" / "sglang" / "kernels" / "aot" / "Dockerfile",
]

README_KEYWORD_GROUPS = [
    ("cache", "缓存"),
    ("chown", "chmod", "go-w"),
    ("non-root", "非默认用户"),
    ("TORCH_EXTENSIONS_DIR",),
]

README_KEYWORDS = (
    "cache",
    "缓存",
    "chown",
    "chmod",
    "go-w",
    "非默认用户",
    "non-root",
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def missing_paths():
    return [str(p) for p in DOCKERFILES + READMES if not p.is_file()]


def test_no_world_writable_permissions():
    for dockerfile in DOCKERFILES:
        content = _read(dockerfile)
        assert not WORLD_WRITABLE.search(content), (
            f"{dockerfile} must not grant world-writable permissions"
        )
        assert not WORKSPACE_WORLD_WRITABLE.search(content), (
            f"{dockerfile} must not make /workspace world-writable"
        )


def test_minimal_permission_statements():
    for dockerfile in HARDENED_DOCKERFILES:
        content = _read(dockerfile)
        assert CHOWN.search(content), f"{dockerfile} should chown the runtime directory"
        assert MIN_PERMISSION.search(content), (
            f"{dockerfile} should remove group/other write access with chmod -R go-w"
        )


def test_devcontainer_workspace_not_world_writable():
    dockerfile = REPO_ROOT / ".devcontainer" / "Dockerfile"
    content = _read(dockerfile)
    assert not WORLD_WRITABLE.search(content), (
        f"{dockerfile} must not grant world-writable permissions"
    )
    assert not WORKSPACE_WORLD_WRITABLE.search(content), (
        f"{dockerfile} must not make /workspace world-writable"
    )


def test_readme_non_root_cache_guidance():
    for readme in READMES:
        content = _read(readme)
        missing = [
            group
            for group in README_KEYWORD_GROUPS
            if not any(keyword in content for keyword in group)
        ]
        assert not missing, (
            f"{readme} should document non-root cache guidance; "
            f"missing keyword groups: {missing}"
        )
        keyword_hits = sum(1 for keyword in README_KEYWORDS if keyword in content)
        assert keyword_hits >= 2, (
            f"{readme} should mention at least two cache/permission keywords, "
            f"found {keyword_hits}"
        )
        assert "```bash" in content, (
            f"{readme} should include an executable command block"
        )
        assert re.search(
            r"^#{2,4}\s+.*(non-root|非默认用户)", content, re.MULTILINE
        ), f"{readme} should have a dedicated non-root section heading"
        assert "TORCH_EXTENSIONS_DIR" in content, (
            f"{readme} should mention TORCH_EXTENSIONS_DIR in the non-root guidance"
        )


CHECKS = [
    test_no_world_writable_permissions,
    test_minimal_permission_statements,
    test_devcontainer_workspace_not_world_writable,
    test_readme_non_root_cache_guidance,
]


def readme_keyword_hit_count() -> int:
    total = 0
    for readme in READMES:
        content = _read(readme)
        total += sum(1 for keyword in README_KEYWORDS if keyword in content)
    return total


def readme_keyword_hits_per_file() -> dict:
    return {
        str(readme.relative_to(REPO_ROOT)): sum(
            1 for keyword in README_KEYWORDS if keyword in _read(readme)
        )
        for readme in READMES
    }
