import re
import sys
from pathlib import Path

# Stdlib-only checks for Dockerfile minimal-permission hardening and README
# non-root cache guidance. This module must never import sglang or pytest at
# module scope so CI can run it with a bare python3 interpreter.
#
# Run directly with `python3 test/dockerfile_permissions/test_dockerfile_permissions.py`
# or via pytest with the sibling pytest.ini.

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
    keyword_hits = sum(1 for keyword in README_KEYWORDS if keyword in content)
    assert keyword_hits >= 2, (
        f"{readme} should mention at least two cache/permission keywords, "
        f"found {keyword_hits}"
    )
    assert "```bash" in content, f"{readme} should include an executable command block"
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


def _missing_paths() -> list:
    paths = [*DOCKERFILES, *READMES]
    return [path for path in paths if not path.is_file()]


def _readme_keyword_hit_count() -> int:
    total = 0
    for readme in READMES:
        content = _read(readme)
        total += sum(1 for keyword in README_KEYWORDS if keyword in content)
    return total


def _readme_keyword_hits_per_file() -> dict:
    return {
        str(readme.relative_to(REPO_ROOT)): sum(
            1 for keyword in README_KEYWORDS if keyword in _read(readme)
        )
        for readme in READMES
    }


def _assert_stdlib_only() -> None:
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
        if top == "pytest":
            continue
        raise AssertionError(
            f"test_dockerfile_permissions must stay stdlib-only; "
            f"unexpected module imported: {name}"
        )


def _run_standalone() -> int:
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
