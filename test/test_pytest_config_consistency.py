import ast
import configparser
import fnmatch
import glob
import os
import subprocess
import sys

try:
    import pytest
except ImportError:  # pragma: no cover - pytest is the runner itself
    pytest = None

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TEST_DIR)
PYTEST_INI = os.path.join(TEST_DIR, "pytest.ini")
REGISTERED_DIR = os.path.join(TEST_DIR, "registered")
RUN_SUITE = os.path.join(TEST_DIR, "run_suite.py")


def _load_pytest_ini():
    parser = configparser.ConfigParser()
    parser.read(PYTEST_INI)
    assert parser.has_section("pytest"), f"missing [pytest] section in {PYTEST_INI}"
    return parser["pytest"]


def _testpaths():
    section = _load_pytest_ini()
    return [
        p.strip() for p in section.get("testpaths", "").splitlines() if p.strip()
    ]


def _python_files():
    section = _load_pytest_ini()
    return [p.strip() for p in section.get("python_files", "").split() if p.strip()]


def _registered_files():
    return [
        f
        for f in glob.glob(os.path.join(REGISTERED_DIR, "**", "*.py"), recursive=True)
        if os.path.basename(f) not in ("conftest.py", "__init__.py")
    ]


def test_pytest_ini_has_discovery_config():
    section = _load_pytest_ini()
    assert section.get("testpaths", "").strip(), "testpaths must be set"
    assert section.get("python_files", "").strip(), "python_files must be set"
    assert section.get("markers", "").strip(), "markers must be declared"


def test_testpaths_matches_run_suite_scan_root():
    testpaths = _testpaths()
    assert testpaths, "testpaths must not be empty"
    # testpaths are resolved relative to the ini file's directory (test/).
    resolved = {os.path.normpath(os.path.join(TEST_DIR, p)) for p in testpaths}
    # run_suite.py scans test/registered/**/*.py; testpaths must cover it.
    assert os.path.normpath(REGISTERED_DIR) in resolved or any(
        os.path.normpath(REGISTERED_DIR).startswith(r + os.sep) for r in resolved
    ), f"testpaths {testpaths} does not cover {REGISTERED_DIR}"


def test_rootdir_is_repo_root():
    section = _load_pytest_ini()
    rootdir = section.get("rootdir", "").strip()
    assert rootdir, "rootdir must be set"
    resolved = os.path.normpath(os.path.join(TEST_DIR, rootdir))
    assert resolved == os.path.normpath(REPO_ROOT), (
        f"rootdir {rootdir!r} resolves to {resolved}, expected {REPO_ROOT}"
    )


def test_python_files_matches_registered_tests():
    patterns = _python_files()
    assert patterns, "python_files must not be empty"
    files = _registered_files()
    if not files:
        if pytest is not None:
            pytest.skip("no registered test files present in this checkout")
        return
    for path in files:
        name = os.path.basename(path)
        assert any(fnmatch.fnmatch(name, pat) for pat in patterns), (
            f"python_files {patterns} does not match registered test {name}"
        )


def test_registered_suite_paths_exist_and_collectable():
    """Every file run_suite.py would discover must exist and be collectable."""
    files = _registered_files()
    assert files, f"no registered test files found under {REGISTERED_DIR}"
    for path in files:
        assert os.path.isfile(path), f"registered suite path missing: {path}"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "no:randomly",
            "--rootdir",
            REPO_ROOT,
            "-c",
            PYTEST_INI,
            REGISTERED_DIR,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_collection_nonzero_from_repo_root_and_test_dir():
    def _collect(cwd):
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "--collect-only",
                "-q",
                "-p",
                "no:randomly",
                "--rootdir",
                REPO_ROOT,
                "-c",
                PYTEST_INI,
            ],
            cwd=cwd,
            capture_output=True,
            text=True,
        )

    root_proc = _collect(REPO_ROOT)
    test_proc = _collect(TEST_DIR)
    assert root_proc.returncode == 0, root_proc.stdout + root_proc.stderr
    assert test_proc.returncode == 0, test_proc.stdout + test_proc.stderr

    def _count(proc):
        for line in reversed(proc.stdout.splitlines()):
            line = line.strip()
            if "test" in line and ("collected" in line or "selected" in line):
                for token in line.split():
                    if token.isdigit():
                        return int(token)
        return 0

    root_count = _count(root_proc)
    test_count = _count(test_proc)
    assert root_count > 0, "no tests collected from repo root"
    assert test_count > 0, "no tests collected from test dir"
    assert root_count == test_count, (
        f"collection mismatch: repo root={root_count}, test dir={test_count}"
    )


def _load_run_suite_ast():
    assert os.path.isfile(RUN_SUITE), f"missing {RUN_SUITE}"
    with open(RUN_SUITE, encoding="utf-8") as f:
        return ast.parse(f.read(), filename=RUN_SUITE)


def _suite_dicts_from_ast(tree):
    suite_dicts = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id.endswith("_SUITES"):
                try:
                    suite_dicts[target.id] = ast.literal_eval(node.value)
                except ValueError:
                    continue
    return suite_dicts


def test_run_suite_suite_registry_resolves():
    """run_suite.py's suite registry must resolve without importing sglang."""
    # Parse the suite dicts by AST so this check needs no sglang runtime and
    # no GPU/torch import chain.
    tree = _load_run_suite_ast()
    suite_dicts = _suite_dicts_from_ast(tree)

    assert suite_dicts, "no *_SUITES registries found in run_suite.py"
    total = 0
    for name, mapping in suite_dicts.items():
        assert isinstance(mapping, dict), f"{name} is not a dict"
        for backend, suites in mapping.items():
            assert isinstance(suites, list), f"{name}[{backend}] is not a list"
            for suite in suites:
                assert isinstance(suite, str) and suite, (
                    f"{name}[{backend}] has invalid suite entry: {suite!r}"
                )
                total += 1
    assert total > 0, "run_suite.py registers no suites"


def test_run_suite_registered_suite_paths_exist():
    """Every suite registered in run_suite.py must resolve to a real scan root."""
    tree = _load_run_suite_ast()
    suite_dicts = _suite_dicts_from_ast(tree)
    assert suite_dicts, "no *_SUITES registries found in run_suite.py"
    # run_suite.py scans test/registered/**/*.py; the scan root must exist and
    # contain at least one discoverable test file for suites to resolve.
    assert os.path.isdir(REGISTERED_DIR), f"missing scan root {REGISTERED_DIR}"
    files = _registered_files()
    assert files, f"no registered test files under {REGISTERED_DIR}"
    patterns = _python_files()
    for path in files:
        name = os.path.basename(path)
        assert any(fnmatch.fnmatch(name, pat) for pat in patterns), (
            f"registered suite path {path} not collectable by python_files {patterns}"
        )


def _legacy_run_suite_suite_registry_resolves():
    with open(RUN_SUITE, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=RUN_SUITE)

    suite_dicts = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id.endswith("_SUITES"):
                try:
                    suite_dicts[target.id] = ast.literal_eval(node.value)
                except ValueError:
                    continue

    assert suite_dicts, "no *_SUITES registries found in run_suite.py"
    total = 0
    for name, mapping in suite_dicts.items():
        assert isinstance(mapping, dict), f"{name} is not a dict"
        for backend, suites in mapping.items():
            assert isinstance(suites, list), f"{name}[{backend}] is not a list"
            for suite in suites:
                assert isinstance(suite, str) and suite, (
                    f"{name}[{backend}] has invalid suite entry: {suite!r}"
                )
                total += 1
    assert total > 0, "run_suite.py registers no suites"


def test_run_suite_hw_mapping_covers_suite_backends():
    """Every backend used in the suite dicts must exist in HW_MAPPING."""
    import ast

    with open(RUN_SUITE, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=RUN_SUITE)

    hw_mapping = None
    suite_backends = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if not isinstance(target, ast.Name):
                continue
            if target.id == "HW_MAPPING":
                try:
                    hw_mapping = ast.literal_eval(node.value)
                except ValueError:
                    hw_mapping = None
            elif target.id.endswith("_SUITES"):
                try:
                    mapping = ast.literal_eval(node.value)
                except ValueError:
                    continue
                if isinstance(mapping, dict):
                    suite_backends.update(mapping.keys())

    assert hw_mapping is not None, "HW_MAPPING not found in run_suite.py"
    assert suite_backends, "no suite backends found in run_suite.py"
    # Suite dicts key on HWBackend members; HW_MAPPING keys are the CLI names.
    # Both must be non-empty and the CLI mapping must cover every backend name
    # that appears as a suite key (compared case-insensitively).
    cli_names = {str(k).lower() for k in hw_mapping}
    for backend in suite_backends:
        name = getattr(backend, "name", str(backend)).lower()
        assert name in cli_names, (
            f"suite backend {name!r} missing from HW_MAPPING {sorted(cli_names)}"
        )
