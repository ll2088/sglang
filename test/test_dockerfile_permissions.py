import re
import sys
from pathlib import Path

# This module intentionally depends only on the Python standard library so the
# permission/documentation checks can run without installing sglang or pytest.

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


def _check_no_world_writable_permissions(dockerfile: Path) -> None:
    content = _read(dockerfile)
    assert not WORLD_WRITABLE.search(content), (
        f"{dockerfile} must not grant world-writable permissions"
    )


def _check_minimal_permission_statements(dockerfile: Path) -> None:
    content = _read(dockerfile)
    assert CHOWN.search(content), f"{dockerfile} should chown the runtime directory"
    assert MIN_PERMISSION.search(content), (
        f"{dockerfile} should remove group/other write access with chmod -R go-w"
    )


def _check_readme_non_root_cache_guidance(readme: Path) -> None:
    content = _read(readme)
    keywords = ["cache", "chown", "chmod", "go-w", "non-root", "TORCH_EXTENSIONS_DIR"]
    hits = sum(1 for keyword in keywords if keyword in content)
    assert hits >= 3, f"{readme} should document non-root cache guidance"
    assert "```bash" in content, f"{readme} should include an executable command block"


def _run_checks() -> int:
    failures = 0
    for check in (
        test_no_world_writable_permissions,
        test_minimal_permission_statements,
        test_readme_non_root_cache_guidance,
    ):
        try:
            check()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {check.__name__}: {exc}", file=sys.stderr)
        else:
            print(f"PASS {check.__name__}")
    return 1 if failures else 0


def test_no_world_writable_permissions() -> None:
    for dockerfile in DOCKERFILES:
        _check_no_world_writable_permissions(dockerfile)


def test_minimal_permission_statements() -> None:
    for dockerfile in DOCKERFILES:
        _check_minimal_permission_statements(dockerfile)


def test_readme_non_root_cache_guidance() -> None:
    for readme in READMES:
        _check_readme_non_root_cache_guidance(readme)


def main() -> int:
    return _run_checks()


if __name__ == "__main__":
    sys.exit(main())
