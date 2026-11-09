import configparser
import os


UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.dirname(UNIT_DIR)
REPO_ROOT = os.path.dirname(TEST_DIR)


def test_unit_pytest_ini_is_self_contained():
    """The unit config must not require plugins beyond pytest itself.

    CI verification for these tests installs only pytest, so any option that
    needs an extra plugin would surface as a collection error rather than a
    test result.
    """
    ini_path = os.path.join(UNIT_DIR, "pytest.ini")
    assert os.path.exists(ini_path)
    parser = configparser.ConfigParser()
    parser.read(ini_path)
    assert parser.has_section("pytest")
    assert parser.get("pytest", "testpaths").strip() == "."
    assert parser.get("pytest", "rootdir").strip() == "."
    # The cache provider must be disabled explicitly so the documented
    # `-p no:cacheprovider` invocation and the ini agree.
    assert "no:cacheprovider" in parser.get("pytest", "addopts")
    # asyncio_mode requires pytest-asyncio; the minimal CI environment only
    # installs pytest, so the unit config must not depend on that plugin.
    assert not parser.has_option("pytest", "asyncio_mode")


def test_unit_launcher_exists_and_is_importable():
    """The dependency-free launcher must exist next to the unit tests."""
    launcher = os.path.join(UNIT_DIR, "run_unit_tests.py")
    assert os.path.exists(launcher)
    with open(launcher, encoding="utf-8") as f:
        text = f.read()
    # The launcher must not import sglang: it is the entry point used when the
    # full runtime is unavailable.
    assert "import sglang" not in text
    assert "pytest.main" in text


def test_unit_launcher_documents_minimal_install():
    """The launcher's missing-pytest diagnostic must name the minimal install."""
    launcher = os.path.join(UNIT_DIR, "run_unit_tests.py")
    with open(launcher, encoding="utf-8") as f:
        text = f.read()
    # The diagnostic must point at a pytest-only install path and must not
    # send CI back to the full sglang dependency resolution.
    assert "pip install pytest" in text
    assert "pip install -e python" not in text


def test_conftest_installs_stubs_before_collection():
    """conftest.py must install the sglang.test.ci stubs at import time."""
    conftest_path = os.path.join(UNIT_DIR, "conftest.py")
    assert os.path.exists(conftest_path)
    with open(conftest_path, encoding="utf-8") as f:
        text = f.read()
    assert "_install_ci_stubs()" in text
    # The call must be at module scope (not inside a fixture) so it runs
    # before pytest imports any test module.
    assert text.rstrip().endswith("_install_ci_stubs()")


def test_readme_exit_code_keywords_present():
    """README must document the three non-zero states the runner distinguishes."""
    readme_path = os.path.join(TEST_DIR, "README.md")
    with open(readme_path, encoding="utf-8") as f:
        text = f.read()
    for keyword in (
        "Collection failure",
        "Test execution failure",
        "No tests matched",
    ):
        assert keyword in text, f"README missing exit-code keyword: {keyword}"


def test_setup_env_guard_precedes_pip_install():
    """The editable guard must run before any pip invocation."""
    script = os.path.join(UNIT_DIR, "setup_env.sh")
    with open(script, encoding="utf-8") as f:
        text = f.read()
    guard_pos = text.find("--editable")
    install_pos = text.find("pip install")
    assert guard_pos != -1
    assert install_pos != -1
    assert guard_pos < install_pos


def test_setup_env_installs_only_minimal_requirements():
    """setup_env.sh must install the pytest-only requirements file."""
    script = os.path.join(UNIT_DIR, "setup_env.sh")
    with open(script, encoding="utf-8") as f:
        text = f.read()
    assert "requirements.txt" in text
    # The full sglang dependency resolution must never be triggered here.
    assert "pip install -e" not in text
    assert "pip install --editable" not in text


def test_verify_entrypoint_exists_and_uses_distinct_exit_codes():
    """verify_run_suite.py must exist and separate env from dimension failure."""
    entry = os.path.join(UNIT_DIR, "verify_run_suite.py")
    assert os.path.exists(entry)
    with open(entry, encoding="utf-8") as f:
        text = f.read()
    assert "EXIT_ENV_FAILURE = 2" in text
    assert "EXIT_DIMENSION_FAILURE = 1" in text
    assert "pytest.main" in text
    assert "import sglang" not in text
