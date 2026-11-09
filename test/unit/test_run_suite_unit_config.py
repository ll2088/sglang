import configparser
import os


UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.dirname(UNIT_DIR)


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
    # asyncio_mode is a pytest-asyncio option; it is harmless when the plugin
    # is absent, but the cache provider must be disabled explicitly so the
    # documented `-p no:cacheprovider` invocation and the ini agree.
    assert "no:cacheprovider" in parser.get("pytest", "addopts")


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
