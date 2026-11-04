import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

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
MIN_PERMISSION = re.compile(r"chmod\s+-R\s+go-w")
CHOWN = re.compile(r"chown\s+-R\s+\S+")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


@pytest.mark.parametrize("dockerfile", DOCKERFILES, ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_no_world_writable_permissions(dockerfile: Path):
    content = _read(dockerfile)
    assert not WORLD_WRITABLE.search(content), (
        f"{dockerfile} must not grant world-writable permissions"
    )


@pytest.mark.parametrize("dockerfile", DOCKERFILES, ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_minimal_permission_statements(dockerfile: Path):
    content = _read(dockerfile)
    assert CHOWN.search(content), f"{dockerfile} should chown the runtime directory"
    assert MIN_PERMISSION.search(content), (
        f"{dockerfile} should remove group/other write access with chmod -R go-w"
    )


@pytest.mark.parametrize("readme", READMES, ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_readme_non_root_cache_guidance(readme: Path):
    content = _read(readme)
    keywords = ["cache", "chown", "chmod", "go-w", "non-root", "TORCH_EXTENSIONS_DIR"]
    hits = sum(1 for keyword in keywords if keyword in content)
    assert hits >= 3, f"{readme} should document non-root cache guidance"
    assert "```bash" in content, f"{readme} should include an executable command block"
