import re
import sys
from pathlib import Path

# This module intentionally depends only on the Python standard library so the
# permission/documentation checks can run without installing sglang or pytest.
# Run directly with `python3 test/test_dockerfile_permissions.py`; pytest is
# optional and only used when the file is collected by a pytest run.

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

# Dockerfiles that must carry the minimal-permission hardening. The acceptance
# grep only forbids world-writable modes in all four files, but the hardening
# statements are expected in the three images that ship runtime/build trees.
HARDENED_DOCKERFILES = [
    REPO_ROOT / "docker" / "Dockerfile",
    REPO_ROOT / "examples" / "frontend_language" / "usage" / "triton" / "Dockerfile",
    REPO_ROOT / "python" / "sglang" / "kernels" / "aot" / "Dockerfile",
]

README_KEYWORDS = [
    "cache",
    "缓存",
    "chown",
    "chmod",
    "go-w",
    "非默认用户",
    "non-root",
    "TORCH_EXTENSIONS_DIR",
]


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
    hits = sum(1 for keyword in README_KEYWORDS if keyword in content)
    assert hits >= 6, (
        f"{readme} should document non-root cache guidance "
        f"(found {hits} keyword hits, need >= 6)"
    )
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


def _missing_paths() -> list[Path]:
    paths = [*DOCKERFILES, *READMES]
    return [path for path in paths if not path.is_file()]


def test_no_world_writable_permissions() -> None:
    for dockerfile in DOCKERFILES:
        _check_no_world_writable_permissions(dockerfile)


def test_minimal_permission_statements() -> None:
    for dockerfile in HARDENED_DOCKERFILES:
        _check_minimal_permission_statements(dockerfile)


def test_readme_non_root_cache_guidance() -> None:
    for readme in READMES:
        _check_readme_non_root_cache_guidance(readme)


def main() -> int:
    missing = _missing_paths()
    if missing:
        for path in missing:
            print(f"FAIL missing expected file: {path}", file=sys.stderr)
        return 1
    return _run_checks()


if __name__ == "__main__":
    sys.exit(main())
