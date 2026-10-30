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
    resolved = {os.path.normpath(os.path.join(TEST_DIR, p)) for p in testpaths}
    # run_suite.py scans test/registered/**/*.py; testpaths must cover it.
    assert os.path.normpath(REGISTERED_DIR) in resolved or any(
        os.path.normpath(REGISTERED_DIR).startswith(r + os.sep) for r in resolved
    ), f"testpaths {testpaths} does not cover {REGISTERED_DIR}"


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
            [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:randomly"],
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
