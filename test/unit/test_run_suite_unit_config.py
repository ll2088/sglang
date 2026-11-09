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


def test_unit_requirements_is_pytest_only():
    """The unit tests must be installable without the full sglang runtime.

    CI environment preparation previously ran `pip install -e python`, which
    resolves the full sglang dependency set and can fail before any test runs.
    The pytest-only requirements file is the documented minimal entry point.
    """
    req_path = os.path.join(UNIT_DIR, "requirements.txt")
    assert os.path.exists(req_path)
    with open(req_path, encoding="utf-8") as f:
        lines = [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]
    assert lines == ["pytest"], lines


def test_unit_launcher_documents_minimal_install():
    """The launcher's missing-pytest diagnostic must name the minimal install."""
    launcher = os.path.join(UNIT_DIR, "run_unit_tests.py")
    with open(launcher, encoding="utf-8") as f:
        text = f.read()
    assert "test/unit/requirements.txt" in text
    assert "pip install pytest" in text


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


def test_verify_run_suite_entrypoint_exists():
    """README documents verify_run_suite.py as the CI entrypoint; it must exist."""
    entrypoint = os.path.join(UNIT_DIR, "verify_run_suite.py")
    assert os.path.exists(entrypoint)
    with open(entrypoint, encoding="utf-8") as f:
        text = f.read()
    # The entrypoint must be pytest-only: no sglang import, and it must expose
    # the infra-vs-test exit-code distinction the README promises.
    assert "import sglang" not in text
    assert "EXIT_INFRA_FAILURE = 2" in text
    assert "EXIT_DIMENSION_FAILURE = 1" in text
    for dimension in (
        "collect-diagnostics",
        "exit-codes",
        "tests",
        "usability",
    ):
        assert dimension in text, f"verify_run_suite missing dimension: {dimension}"
    # The entrypoint must be runnable without the full sglang dependency set:
    # pytest is imported lazily so `--help` and the environment banner work
    # even when pytest is missing (that case exits 2, not 1).
    assert "import pytest" in text
    assert "NOT INSTALLED" in text


def test_verify_run_suite_importable_without_pytest_runtime():
    """The entrypoint must import cleanly (pytest is imported lazily)."""
    import importlib.util

    entrypoint = os.path.join(UNIT_DIR, "verify_run_suite.py")
    spec = importlib.util.spec_from_file_location(
        "verify_run_suite_under_test", entrypoint
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.EXIT_OK == 0
    assert module.EXIT_DIMENSION_FAILURE == 1
    assert module.EXIT_INFRA_FAILURE == 2
    assert len(module.DIMENSIONS) == 4
    # Every dimension must point at existing test files so a CI job cannot
    # silently run zero tests for a dimension.
    for name, targets in module.DIMENSIONS.items():
        assert targets, f"dimension {name} has no targets"
        for target in targets:
            assert os.path.exists(target), f"{name}: missing {target}"


def test_verify_run_suite_dimensions_cover_required_checks():
    """The four dimensions must cover the acceptance checks from the plan."""
    import importlib.util

    entrypoint = os.path.join(UNIT_DIR, "verify_run_suite.py")
    spec = importlib.util.spec_from_file_location(
        "verify_run_suite_dimensions_under_test", entrypoint
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    exit_codes = module.DIMENSIONS["exit-codes"]
    assert any(
        os.path.basename(p) == "test_run_suite_exit_codes.py" for p in exit_codes
    )
    collect = module.DIMENSIONS["collect-diagnostics"]
    assert any(
        os.path.basename(p) == "test_run_suite_collect_diagnostics.py"
        for p in collect
    )
    usability = module.DIMENSIONS["usability"]
    assert any(
        os.path.basename(p) == "test_run_suite_unit_config.py" for p in usability
    )
