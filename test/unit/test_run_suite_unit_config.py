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
    assert "test/unit/setup_env.sh" in text
    assert "test/unit/requirements.txt" in text


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


def test_setup_env_script_is_pytest_only():
    """The lightweight setup script must not install the full sglang deps.

    CI environment preparation previously ran `pip install -e python`, which
    resolves the full sglang dependency set (torch, kernels, ...) and can fail
    before any test runs. The setup script must install only
    test/unit/requirements.txt and self-check pytest.
    """
    script = os.path.join(UNIT_DIR, "setup_env.sh")
    assert os.path.exists(script)
    with open(script, encoding="utf-8") as f:
        text = f.read()
    assert "test/unit/requirements.txt" in text
    # Must not trigger full dependency resolution.
    assert "pip install -e python" not in text
    assert "pip install -e ./python" not in text
    # Must self-check pytest so env failures are distinguishable.
    assert "import pytest" in text
    assert "pytest --version" in text


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
    # The entrypoint must not shell out to a full sglang install; it is the
    # lightweight path CI uses instead of `pip install -e python`.
    assert "pip install -e python" not in text
    # The entrypoint must point at the pytest-only setup entry so CI does not
    # fall back to resolving the full sglang dependency set.
    assert "test/unit/setup_env.sh" in text


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
    tests = module.DIMENSIONS["tests"]
    assert any(
        os.path.basename(p) == "test_run_suite_exit_codes.py" for p in tests
    )


def test_setup_env_script_is_executable_and_guarded():
    """setup_env.sh must fail fast and self-check pytest."""
    script = os.path.join(UNIT_DIR, "setup_env.sh")
    with open(script, encoding="utf-8") as f:
        text = f.read()
    assert text.startswith("#!/usr/bin/env bash")
    assert "set -euo pipefail" in text
    assert "import pytest" in text
    assert "pytest --version" in text
    # The guard must reject the full sglang dependency set coming back.
    assert "sglang==" in text
    assert "quack-kernels" in text
    # The script must print the resolved path so CI logs prove the lightweight
    # entry was hit instead of a full `pip install -e python`.
    assert "BASH_SOURCE" in text
    assert "[setup] using" in text
    # It must install only the minimal requirements file.
    assert "-r \"${REQUIREMENTS}\"" in text
    # It must be executable so CI can invoke it directly.
    assert os.access(script, os.X_OK), "setup_env.sh must be executable"


def test_setup_env_script_has_dependency_smoke_guard():
    """setup_env.sh must assert the full sglang deps were not resolved."""
    script = os.path.join(UNIT_DIR, "setup_env.sh")
    with open(script, encoding="utf-8") as f:
        text = f.read()
    # The smoke guard must grep the install log for the full dependency
    # resolution pattern and exit non-zero when it appears.
    assert "Collecting .*from sglang==" in text
    assert "grep -Eq" in text
    assert "exit 2" in text
    # The install log must be captured so the guard can inspect it.
    assert "tee" in text
    assert "LOG_FILE" in text


def test_setup_env_scripts_reject_editable_before_install():
    """The editable guard must run before any pip invocation.

    CI previously failed during environment preparation because the full
    sglang dependency set was resolved. The guard must therefore appear before
    the pip install line in both setup entries, so an accidental editable
    install is rejected with exit 2 instead of continuing.
    """
    for script in (
        os.path.join(REPO_ROOT, "setup_env.sh"),
        os.path.join(UNIT_DIR, "setup_env.sh"),
    ):
        with open(script, encoding="utf-8") as f:
            text = f.read()
        guard_pos = text.find("--editable")
        install_pos = text.find("pip install")
        assert guard_pos != -1, f"{script} missing editable guard"
        assert install_pos != -1, f"{script} missing pip install"
        assert guard_pos < install_pos, (
            f"{script}: editable guard must precede pip install"
        )


def test_setup_env_scripts_print_resolved_path():
    """Both setup entries must print the resolved script path.

    CI logs must prove the lightweight entry was hit instead of a full
    `pip install -e python`; the `[setup] using` banner is the marker.
    """
    for script in (
        os.path.join(REPO_ROOT, "setup_env.sh"),
        os.path.join(UNIT_DIR, "setup_env.sh"),
    ):
        with open(script, encoding="utf-8") as f:
            text = f.read()
        assert "[setup] using" in text, script
        assert "BASH_SOURCE" in text, script


def test_root_setup_env_delegates_to_lightweight_script():
    """The repo-root setup entry must delegate to the pytest-only script.

    CI environment preparation previously ran `pip install -e python`, which
    resolves the full sglang dependency set and fails before any test runs.
    Whichever setup path CI calls, it must land on the lightweight script.
    """
    root_script = os.path.join(REPO_ROOT, "setup_env.sh")
    assert os.path.exists(root_script)
    with open(root_script, encoding="utf-8") as f:
        text = f.read()
    assert text.startswith("#!/usr/bin/env bash")
    assert "set -euo pipefail" in text
    assert "test/unit/setup_env.sh" in text
    assert "BASH_SOURCE" in text
    assert "[setup] using" in text
    # Must not trigger full dependency resolution itself.
    assert "pip install -e python" not in text
    assert "pip install -e ./python" not in text
    assert os.access(root_script, os.X_OK), "setup_env.sh must be executable"


def test_root_setup_env_rejects_editable_install():
    """The repo-root entry must reject editable installs explicitly."""
    root_script = os.path.join(REPO_ROOT, "setup_env.sh")
    with open(root_script, encoding="utf-8") as f:
        text = f.read()
    assert "--editable" in text
    assert "exit 2" in text
    # It must not itself resolve the full sglang dependency set.
    assert "pip install -e" not in text
    assert "sglang==" not in text


def test_setup_env_scripts_are_cwd_independent():
    """Both setup entries must resolve paths from their own location.

    CI may invoke the script from an arbitrary working directory; a relative
    path would silently miss the lightweight entry and fall back to the full
    `pip install -e python` resolution.
    """
    for script in (
        os.path.join(REPO_ROOT, "setup_env.sh"),
        os.path.join(UNIT_DIR, "setup_env.sh"),
    ):
        with open(script, encoding="utf-8") as f:
            text = f.read()
        assert "BASH_SOURCE" in text, f"{script} must resolve its own dir"
        assert "SCRIPT_DIR" in text, f"{script} must use an absolute dir"


def test_setup_env_scripts_reject_editable_install():
    """Neither setup entry may shell out to an editable sglang install."""
    for script in (
        os.path.join(REPO_ROOT, "setup_env.sh"),
        os.path.join(UNIT_DIR, "setup_env.sh"),
    ):
        with open(script, encoding="utf-8") as f:
            text = f.read()
        assert "pip install -e" not in text, f"{script} must not editable-install"
        assert "-e python" not in text, f"{script} must not editable-install"
        # The guard must fail fast (non-zero) when an editable arg is passed,
        # so a CI entry that misses the lightweight path cannot silently fall
        # back to resolving the full sglang dependency set.
        assert "--editable" in text, f"{script} must reject --editable"
        assert "exit 2" in text, f"{script} must exit non-zero on editable"


def test_setup_env_scripts_do_not_resolve_sglang_deps():
    """The setup entries must not trigger sglang dependency resolution.

    CI logs previously showed py-spy / pybase64 / quack-kernels being
    downloaded during environment preparation, which means the lightweight
    entry was not hit. The scripts must only install the pytest-only
    requirements file.
    """
    for script in (
        os.path.join(REPO_ROOT, "setup_env.sh"),
        os.path.join(UNIT_DIR, "setup_env.sh"),
    ):
        with open(script, encoding="utf-8") as f:
            text = f.read()
        assert "requirements.txt" in text, f"{script} must install minimal reqs"
        assert "sglang==" not in text, f"{script} must not pin sglang"
        assert "quack-kernels" not in text, f"{script} must not pin kernels"
        assert "py-spy" not in text, f"{script} must not pin py-spy"


def test_setup_env_scripts_are_executable():
    """Both setup entries must be directly invocable by CI."""
    for script in (
        os.path.join(REPO_ROOT, "setup_env.sh"),
        os.path.join(UNIT_DIR, "setup_env.sh"),
    ):
        assert os.path.exists(script), script
        assert os.access(script, os.X_OK), f"{script} must be executable"
        with open(script, encoding="utf-8") as f:
            text = f.read()
        assert text.startswith("#!/usr/bin/env bash"), script
        assert "set -euo pipefail" in text, script


def test_verify_run_suite_help_works_without_pytest():
    """--help must work even when pytest is missing (exit 0, no import)."""
    import subprocess
    import sys as _sys

    entrypoint = os.path.join(UNIT_DIR, "verify_run_suite.py")
    proc = subprocess.run(
        [_sys.executable, entrypoint, "--help"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    for dimension in ("collect-diagnostics", "exit-codes", "tests", "usability"):
        assert dimension in proc.stdout
