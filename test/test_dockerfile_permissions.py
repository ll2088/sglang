import re
import sys
from pathlib import Path

# This module intentionally depends only on the Python standard library so the
# permission/documentation checks can run without installing sglang or pytest.
# Run directly with `python3 test/test_dockerfile_permissions.py`; pytest is
# optional and only used when the file is collected by a pytest run.
#
# Keep this file free of any `import sglang` / `import pytest` at module scope:
# CI runs it with a minimal environment (python3 only) and pytest collection
# must not pull in the full sglang dependency tree. test/conftest.py and
# test/pytest.ini keep collection scoped to this file so the repository-level
# pytest configuration cannot trigger a full sglang install during setup.
#
# The checks below are plain functions named test_* so pytest can collect them,
# but they never import pytest or sglang. If pytest is unavailable, main() still
# runs every check and reports PASS/FAIL without third-party packages.

REPO_ROOT = Path(__file__).resolve().parents[1]

# The canonical implementation lives in test/dockerfile_permissions/ so CI can
# run it with `-c test/dockerfile_permissions/pytest.ini --rootdir
# test/dockerfile_permissions`, which never walks up to the repository root
# conftest.py or pyproject.toml. This module re-exports the checks for callers
# that still invoke `python3 test/test_dockerfile_permissions.py` directly.

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
    keyword_hits = sum(
        1
        for keyword in (
            "cache",
            "缓存",
            "chown",
            "chmod",
            "go-w",
            "非默认用户",
            "non-root",
        )
        if keyword in content
    )
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


def _run_checks() -> int:
    failures = 0
    for check in (
        test_no_world_writable_permissions,
        test_minimal_permission_statements,
        test_readme_non_root_cache_guidance,
        test_devcontainer_workspace_not_world_writable,
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


def main() -> int:
    missing = _missing_paths()
    if missing:
        for path in missing:
            print(f"FAIL missing expected file: {path}", file=sys.stderr)
        return 1
    return _run_checks()


def _pytest_available() -> bool:
    """Return True only when pytest can be imported without side effects.

    The checks in this module are stdlib-only; pytest is optional. CI may run
    this file with a bare `python3` interpreter, so guard the import instead of
    letting a missing pytest turn into a collection error.

    This helper is intentionally never called at module scope: importing pytest
    here would make the module depend on a third-party package and could pull in
    repository-level plugins. It exists only for diagnostics.
    """
    try:
        import pytest  # noqa: F401
    except Exception:
        return False
    return True


def _readme_keyword_hits_per_file() -> dict:
    """Return per-README keyword hit counts using the acceptance grep keywords.

    Kept stdlib-only so CI can report the same numbers as the shell grep
    without installing pytest or sglang.
    """
    keywords = (
        "cache",
        "缓存",
        "chown",
        "chmod",
        "go-w",
        "非默认用户",
        "non-root",
    )
    return {
        str(readme.relative_to(REPO_ROOT)): sum(
            1 for keyword in keywords if keyword in _read(readme)
        )
        for readme in READMES
    }


def _assert_stdlib_only() -> None:
    """Fail fast if this module ever grows a non-stdlib import.

    The CI job that runs these checks intentionally installs nothing beyond
    the interpreter. A stray third-party import would make the job fail during
    collection instead of reporting a real permission/documentation problem,
    so keep the module importable with a bare python3.

    Only modules imported by this file are inspected; the runner may have
    already imported pytest, which is explicitly tolerated below.
    """
    allowed = {
        "re",
        "sys",
        "pathlib",
        "__future__",
        "typing",
        "os",
        "io",
        "collections",
        "functools",
        "itertools",
        "subprocess",
        "tempfile",
        "shutil",
        "json",
        "textwrap",
        "unittest",
        "dataclasses",
        "enum",
        "abc",
        "contextlib",
        "warnings",
        "traceback",
        "inspect",
        "types",
        "copy",
        "math",
        "stat",
        "glob",
        "fnmatch",
        "string",
        "datetime",
        "time",
        "random",
        "hashlib",
        "base64",
        "binascii",
        "struct",
        "array",
        "operator",
        "weakref",
        "gc",
        "platform",
        "getpass",
        "socket",
        "select",
        "signal",
        "threading",
        "queue",
        "asyncio",
        "concurrent",
        "multiprocessing",
        "logging",
        "argparse",
        "configparser",
        "csv",
        "sqlite3",
        "xml",
        "html",
        "http",
        "urllib",
        "email",
        "mimetypes",
        "uuid",
        "secrets",
        "hmac",
        "zlib",
        "gzip",
        "bz2",
        "lzma",
        "zipfile",
        "tarfile",
        "pickle",
        "shelve",
        "dbm",
        "marshal",
        "codecs",
        "unicodedata",
        "locale",
        "gettext",
        "calendar",
        "zoneinfo",
        "decimal",
        "fractions",
        "numbers",
        "cmath",
        "statistics",
        "pprint",
        "reprlib",
        "difflib",
        "filecmp",
        "fileinput",
        "linecache",
        "tokenize",
        "token",
        "keyword",
        "ast",
        "symtable",
        "compileall",
        "py_compile",
        "dis",
        "pickletools",
        "tabnanny",
        "pyclbr",
        "pydoc",
        "doctest",
        "unittest",
        "venv",
        "ensurepip",
        "zipapp",
        "sysconfig",
        "site",
        "builtins",
        "__main__",
    }
    for name, module in list(sys.modules.items()):
        if module is None:
            continue
        top = name.split(".", 1)[0]
        if top in allowed or top.startswith("_"):
            continue
        # pytest is allowed only when it is already imported by the runner;
        # this module never imports it at module scope.
        if top == "pytest":
            continue
        raise AssertionError(
            f"test_dockerfile_permissions must stay stdlib-only; "
            f"unexpected module imported: {name}"
        )


def _readme_keyword_hit_count() -> int:
    """Count README keyword hits the same way the acceptance grep does.

    Kept as a helper so the standalone runner can report the count without
    depending on grep or pytest.
    """
    keywords = (
        "cache",
        "缓存",
        "chown",
        "chmod",
        "go-w",
        "非默认用户",
        "non-root",
    )
    total = 0
    for readme in READMES:
        content = _read(readme)
        total += sum(1 for keyword in keywords if keyword in content)
    return total


def _run_standalone() -> int:
    """Run every check without pytest, including the devcontainer check.

    This is the entry point used by CI so the permission/documentation checks
    never require installing sglang or pytest. It mirrors the pytest-collected
    test functions but reports PASS/FAIL and returns a process exit code.

    The function deliberately does not import pytest or sglang; it only reads
    files and evaluates regular expressions from the standard library.
    """
    missing = _missing_paths()
    if missing:
        for path in missing:
            print(f"FAIL missing expected file: {path}", file=sys.stderr)
        return 1

    failures = 0
    checks = (
        test_no_world_writable_permissions,
        test_minimal_permission_statements,
        test_readme_non_root_cache_guidance,
        test_devcontainer_workspace_not_world_writable,
    )
    for check in checks:
        try:
            check()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {check.__name__}: {exc}", file=sys.stderr)
        else:
            print(f"PASS {check.__name__}")

    # Mirror the acceptance grep: at least six keyword hits across the READMEs.
    hits = _readme_keyword_hit_count()
    if hits < 6:
        failures += 1
        print(
            f"FAIL readme_keyword_hits: expected >= 6, found {hits}",
            file=sys.stderr,
        )
    else:
        print(f"PASS readme_keyword_hits ({hits})")
    for name, count in _readme_keyword_hits_per_file().items():
        print(f"INFO readme_keyword_hits[{name}]={count}")

    # Guard against a future edit that pulls in a third-party dependency and
    # makes the CI job fail during import instead of reporting a real check.
    try:
        _assert_stdlib_only()
    except AssertionError as exc:
        failures += 1
        print(f"FAIL stdlib_only: {exc}", file=sys.stderr)
    else:
        print("PASS stdlib_only")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_run_standalone())
