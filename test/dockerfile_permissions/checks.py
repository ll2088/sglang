import re
from pathlib import Path

# Stdlib-only implementation of the Dockerfile permission and README guidance
# checks. Nothing in this module may import sglang or pytest at module scope:
# CI runs it with a bare python3 interpreter so the checks never depend on the
# full sglang dependency tree.

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
# World-writable grants that specifically target the runtime workspace.
WORKSPACE_WORLD_WRITABLE = re.compile(
    r"chmod\s+(-R\s+)?(a\+w|ugo\+w|o\+w|777)\s+[^\n]*/workspace"
)
MIN_PERMISSION = re.compile(r"chmod\s+-R\s+go-w")
CHOWN = re.compile(r"chown\s+-R\s+\S+")

# Dockerfiles that must carry the minimal-permission hardening. The acceptance
# grep only forbids world-writable modes in all four files, but the hardening
# statements are expected in the three images that ship runtime/build trees.
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


def _check_no_world_writable_permissions(dockerfile: Path) -> None:
    content = _read(dockerfile)
    assert not WORLD_WRITABLE.search(content), (
        f"{dockerfile} must not grant world-writable permissions"
    )
    assert not WORKSPACE_WORLD_WRITABLE.search(content), (
        f"{dockerfile} must not make /workspace world-writable"
    )


def _check_minimal_permission_statements(dockerfile: Path) -> None:
    content = _read(dockerfile)
    assert CHOWN.search(content), f"{dockerfile} should chown the runtime directory"
    assert MIN_PERMISSION.search(content), (
        f"{dockerfile} should remove group/other write access with chmod -R go-w"
    )


def _check_readme_non_root_cache_guidance(readme: Path) -> None:
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
    # The acceptance check greps for these keywords and requires at least six
    # hits across the three READMEs; assert the per-file contribution here so a
    # regression is caught without shelling out to grep.
    keyword_hits = sum(1 for keyword in README_KEYWORDS if keyword in content)
    assert keyword_hits >= 2, (
        f"{readme} should mention at least two cache/permission keywords, "
        f"found {keyword_hits}"
    )
    assert "```bash" in content, f"{readme} should include an executable command block"
    # The guidance must be a structured section, not a stray sentence.
    assert re.search(r"^#{2,4}\s+.*(non-root|非默认用户)", content, re.MULTILINE), (
        f"{readme} should have a dedicated non-root section heading"
    )
    assert "TORCH_EXTENSIONS_DIR" in content, (
        f"{readme} should mention TORCH_EXTENSIONS_DIR in the non-root guidance"
    )


def test_no_world_writable_permissions() -> None:
    for dockerfile in DOCKERFILES:
        _check_no_world_writable_permissions(dockerfile)


def test_minimal_permission_statements() -> None:
    for dockerfile in HARDENED_DOCKERFILES:
        _check_minimal_permission_statements(dockerfile)


def test_readme_non_root_cache_guidance() -> None:
    for readme in READMES:
        _check_readme_non_root_cache_guidance(readme)


def test_devcontainer_workspace_not_world_writable() -> None:
    """The devcontainer image must not leave /sgl-workspace world-writable."""
    devcontainer = REPO_ROOT / ".devcontainer" / "Dockerfile"
    content = _read(devcontainer)
    assert not WORLD_WRITABLE.search(content), (
        f"{devcontainer} must not grant world-writable permissions"
    )
    assert not WORKSPACE_WORLD_WRITABLE.search(content), (
        f"{devcontainer} must not make /sgl-workspace world-writable"
    )
    assert CHOWN.search(content), f"{devcontainer} should chown the runtime directory"
    assert MIN_PERMISSION.search(content), (
        f"{devcontainer} should remove group/other write access with chmod -R go-w"
    )


def readme_keyword_hit_count() -> int:
    """Count README keyword hits the same way the acceptance grep does."""
    total = 0
    for readme in READMES:
        content = _read(readme)
        total += sum(1 for keyword in README_KEYWORDS if keyword in content)
    return total


def readme_keyword_hits_per_file() -> dict:
    """Return per-README keyword hit counts using the acceptance grep keywords."""
    return {
        str(readme.relative_to(REPO_ROOT)): sum(
            1 for keyword in README_KEYWORDS if keyword in _read(readme)
        )
        for readme in READMES
    }


def missing_paths() -> list:
    paths = [*DOCKERFILES, *READMES]
    return [path for path in paths if not path.is_file()]


CHECKS = (
    test_no_world_writable_permissions,
    test_minimal_permission_statements,
    test_readme_non_root_cache_guidance,
    test_devcontainer_workspace_not_world_writable,
)
