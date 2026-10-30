import configparser
import os
import subprocess
import sys

import pytest

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TEST_DIR)
PYTEST_INI = os.path.join(TEST_DIR, "pytest.ini")
REGISTERED_DIR = os.path.join(TEST_DIR, "registered")


def _load_pytest_ini():
    parser = configparser.ConfigParser()
    parser.read(PYTEST_INI)
    assert parser.has_section("pytest"), f"missing [pytest] section in {PYTEST_INI}"
    return parser["pytest"]


def test_pytest_ini_has_discovery_config():
    section = _load_pytest_ini()
    assert section.get("testpaths", "").strip(), "testpaths must be set"
    assert section.get("python_files", "").strip(), "python_files must be set"
    assert section.get("markers", "").strip(), "markers must be declared"


def test_testpaths_matches_run_suite_scan_root():
    section = _load_pytest_ini()
    testpaths = [p.strip() for p in section.get("testpaths", "").splitlines() if p.strip()]
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
    section = _load_pytest_ini()
    patterns = [p.strip() for p in section.get("python_files", "").split() if p.strip()]
    assert patterns, "python_files must not be empty"
    sample = os.path.join(REGISTERED_DIR, "core", "test_srt_endpoint.py")
    if not os.path.exists(sample):
        pytest.skip("sample registered test not present in this checkout")
    name = os.path.basename(sample)
    assert any(
        __import__("fnmatch").fnmatch(name, pat) for pat in patterns
    ), f"python_files {patterns} does not match {name}"


def test_collection_nonzero_from_repo_root_and_test_dir():
    def _collect(cwd):
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
            ],
            cwd=cwd,
            capture_output=True,
            text=True,
        )
        return proc

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


def test_registered_suite_paths_exist_and_collectable():
    """Every file run_suite.py would discover must exist and be collectable."""
    import glob

    files = [
        f
        for f in glob.glob(
            os.path.join(REGISTERED_DIR, "**", "*.py"), recursive=True
        )
        if os.path.basename(f) not in ("conftest.py", "__init__.py")
    ]
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
