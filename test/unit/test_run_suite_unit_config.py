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
    assert "test/unit/requirements.txt" in text
    assert "setup_env.sh" in text
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
    # The guard must reject editable installs explicitly rather than silently
    # falling through to the full sglang dependency resolution.
    assert "exit 2" in text
    # The guard must match the short form too, so `setup_env.sh -e` cannot
    # slip past into the full dependency resolution.
    assert "-e|--editable" in text
    # The guard must run before the requirements install so an editable flag
    # can never reach pip.
    assert text.find("-e|--editable") < text.find("pip install -r")


def test_root_setup_env_delegates_to_unit_setup():
    """The root setup entry must delegate to the pytest-only unit script."""
    script = os.path.join(REPO_ROOT, "setup_env.sh")
    assert os.path.exists(script)
    with open(script, encoding="utf-8") as f:
        text = f.read()
    assert "test/unit/setup_env.sh" in text
    # The root entry must never resolve the full sglang dependency set.
    assert "pip install -e" not in text
    assert "pip install --editable" not in text
    # Delegation must resolve paths from the script's own directory so CI can
    # invoke it from any CWD, and must exec the unit script rather than
    # duplicating install logic.
    assert "BASH_SOURCE" in text
    assert "SCRIPT_DIR" in text
    assert "exec bash" in text
    # Fail fast and surface the resolved paths so CI logs prove the lightweight
    # entry was hit instead of a full editable install.
    assert "set -euo pipefail" in text
    assert "[setup]" in text
    # Any editable flag must be rejected before delegation.
    assert "--editable" in text
    assert "exit 2" in text
    # The guard must run before the delegate is exec'd, otherwise CI could
    # still reach the full dependency resolution.
    guard_pos = text.find("--editable")
    exec_pos = text.find("exec bash")
    assert guard_pos != -1
    assert exec_pos != -1
    assert guard_pos < exec_pos
    # The root entry must not install anything itself: only the unit script
    # owns the pytest-only requirements install.
    assert "pip install" not in text


def test_setup_env_installs_only_minimal_requirements():
    """setup_env.sh must install the pytest-only requirements file."""
    script = os.path.join(UNIT_DIR, "setup_env.sh")
    with open(script, encoding="utf-8") as f:
        text = f.read()
    assert "requirements.txt" in text
    # The full sglang dependency resolution must never be triggered here.
    assert "pip install -e" not in text
    assert "pip install --editable" not in text
    # The script must self-check pytest so an environment problem is reported
    # as an environment failure instead of a test failure.
    assert "import pytest" in text
    assert "pytest --version" in text


def test_root_setup_env_rejects_editable_flags():
    """The root entry must reject -e/--editable before delegating."""
    script = os.path.join(REPO_ROOT, "setup_env.sh")
    with open(script, encoding="utf-8") as f:
        text = f.read()
    # Both short and long editable forms must be matched by the guard.
    assert "-e|--editable" in text
    # The guard must run before the delegate is exec'd.
    guard_pos = text.find("-e|--editable")
    exec_pos = text.find("exec bash")
    assert guard_pos != -1
    assert exec_pos != -1
    assert guard_pos < exec_pos
    # The rejection must be actionable: name the pytest-only entry and exit
    # non-zero so CI cannot fall through to the full dependency resolution.
    assert "test/unit/setup_env.sh" in text
    assert "exit 2" in text


def test_setup_env_does_not_resolve_sglang_dependencies():
    """Neither setup entry may trigger the full sglang dependency resolution."""
    for script in (
        os.path.join(REPO_ROOT, "setup_env.sh"),
        os.path.join(UNIT_DIR, "setup_env.sh"),
    ):
        with open(script, encoding="utf-8") as f:
            text = f.read()
        assert "pip install -e" not in text
        assert "pip install --editable" not in text
        assert "pip install -e python" not in text
        # The only install target is the pytest-only requirements file.
        assert "requirements.txt" in text


def test_unit_setup_env_is_standalone_and_pytest_only():
    """The unit setup script must be runnable without the sglang runtime."""
    script = os.path.join(UNIT_DIR, "setup_env.sh")
    with open(script, encoding="utf-8") as f:
        text = f.read()
    # It must resolve its own directory so it works from any CWD.
    assert "BASH_SOURCE" in text
    assert "SCRIPT_DIR" in text
    # It must install only the pytest-only requirements file.
    assert "pip install -r" in text
    assert "requirements.txt" in text
    # It must self-check pytest so an environment problem is reported as an
    # environment failure rather than a test failure.
    assert "import pytest" in text
    assert "pytest --version" in text
    # It must never import or install the sglang runtime.
    assert "import sglang" not in text
    assert "pip install -e" not in text


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
    # All four verification dimensions must be wired into the entrypoint so CI
    # can run them through one command instead of ad-hoc pytest invocations.
    for dimension in (
        "collect-diagnostics",
        "exit-codes",
        "tests",
        "usability",
    ):
        assert dimension in text, f"verify entry missing dimension: {dimension}"
