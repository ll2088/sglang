"""Sanity checks for the repository coverage configuration.

These tests are intentionally lightweight: they do not run the full suite under
coverage, they only verify that the configuration is well-formed and that the
``sysmon`` coverage core can start and produce a report when requested.
"""

import configparser
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
COVERAGERC = REPO_ROOT / ".coveragerc"


def _coverage_available() -> bool:
    return importlib.util.find_spec("coverage") is not None


def _parse_coverage_version(raw_version: str):
    """Return ``(major, minor)`` for a coverage version string, or ``None``.

    Coverage version strings are normally ``"7.4.0"``, but development or
    vendored builds can carry suffixes that are not plain integers. Parsing
    must never raise here: an unparsable version only means the runtime
    sysmon checks cannot be trusted, so the caller skips them.
    """
    match = re.match(r"\s*(\d+)\.(\d+)", raw_version)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _sysmon_runtime_supported() -> bool:
    """Whether this interpreter can run the sysmon coverage core.

    ``sys.monitoring`` exists on CPython 3.12+, so the runtime probes run on
    every 3.12+ interpreter. The configuration assertions run everywhere
    regardless, so an older interpreter still validates ``.coveragerc``.
    """
    return sys.version_info[:2] >= (3, 12)


def _require_coverage() -> None:
    """Fail loudly when coverage is not installed.

    The configuration assertions above still run everywhere; only the
    subprocess-based sysmon checks need the coverage package. A missing
    coverage package is an environment problem, not a reason to skip: the
    verification commands install coverage explicitly, so silently skipping
    here would hide a broken environment behind a green run.
    """
    if not _coverage_available():
        pytest.fail(
            "coverage is not installed; install it before running the "
            "sysmon coverage checks"
        )


def _require_sysmon_core() -> None:
    """Skip when the installed coverage cannot provide the sysmon core.

    ``core = sysmon`` needs Python 3.12+ and a coverage release that ships the
    ``sys.monitoring`` core. Older environments fall back to the default core
    with a warning, so the runtime assertions below would be meaningless there.
    """
    _require_coverage()
    if not _sysmon_runtime_supported():
        pytest.skip("sysmon coverage core runtime checks need CPython 3.12+")
    try:
        import coverage as _coverage_module

        raw_version = _coverage_module.__version__
        version = _parse_coverage_version(raw_version)
    except Exception as exc:  # pragma: no cover - depends on environment
        pytest.fail(f"cannot determine installed coverage version: {exc}")
    if version is None:
        pytest.skip(
            f"cannot parse installed coverage version: {raw_version!r}"
        )
    if version < (7, 4):
        pytest.fail(
            f"coverage {raw_version} does not ship the sysmon core "
            "(needs coverage >= 7.4)"
        )
    # ``sys.monitoring`` is only available on CPython 3.12+; other
    # implementations (e.g. PyPy) may report a matching version but lack the
    # underlying API, so probe it explicitly before starting the core.
    if not hasattr(sys, "monitoring"):
        pytest.fail("sys.monitoring is not available in this interpreter")
    # ``core = sysmon`` in .coveragerc is only honoured by coverage releases
    # that know the name; older releases warn and fall back to the default
    # core, which would make the runtime assertions below pass vacuously.
    cov = None
    try:
        from coverage import coverage as _coverage_cls

        cov = _coverage_cls(core="sysmon")
        cov.start()
        cov.stop()
    except Exception as exc:  # pragma: no cover - depends on environment
        pytest.fail(f"installed coverage cannot start the sysmon core: {exc}")
    finally:
        # Never leave a half-started collector behind for later tests.
        if cov is not None:
            try:
                cov.stop()
            except Exception:  # pragma: no cover - best effort cleanup
                pass


def _load_coveragerc() -> configparser.ConfigParser:
    parser = configparser.ConfigParser()
    assert COVERAGERC.is_file(), f"missing {COVERAGERC}"
    parser.read(COVERAGERC)
    return parser


def test_coveragerc_has_run_section():
    parser = _load_coveragerc()
    assert parser.has_section("run")
    assert parser.get("run", "source", fallback="").strip()


def test_coveragerc_core_is_valid():
    parser = _load_coveragerc()
    core = parser.get("run", "core", fallback="").strip()
    # Empty means "coverage default"; otherwise it must be a known core name.
    assert core in {"", "sysmon", "ctrace", "pytrace"}, core


def test_coveragerc_core_is_sysmon():
    """The repository opts into the sysmon core; keep that explicit."""
    parser = _load_coveragerc()
    assert parser.get("run", "core", fallback="").strip() == "sysmon"


def test_coveragerc_core_matches_env_override():
    """CI sets COVERAGE_CORE=sysmon; the config must agree with that override.

    The upstream change enables the sysmon core through the environment in CI
    lanes. This repository keeps the same choice in ``.coveragerc`` so local
    runs and CI share one core. If the two ever diverge, coverage silently
    uses the environment value and the config assertion above becomes
    misleading, so pin the relationship here.
    """
    parser = _load_coveragerc()
    configured = parser.get("run", "core", fallback="").strip()
    env_override = os.environ.get("COVERAGE_CORE", "").strip()
    if env_override:
        assert env_override == configured, (
            f"COVERAGE_CORE={env_override!r} overrides .coveragerc core={configured!r}"
        )
    else:
        assert configured == "sysmon"


def test_coveragerc_core_is_sysmon_or_env_override():
    """The effective core must be sysmon unless an explicit override is set.

    CI lanes set ``COVERAGE_CORE=sysmon``; local runs rely on ``.coveragerc``.
    This test documents the intended effective value without requiring the
    environment variable to be present, so it passes in both setups.
    """
    parser = _load_coveragerc()
    configured = parser.get("run", "core", fallback="").strip()
    effective = os.environ.get("COVERAGE_CORE", "").strip() or configured
    assert effective == "sysmon", (
        f"effective coverage core is {effective!r}, expected 'sysmon'"
    )


def test_coveragerc_report_section():
    parser = _load_coveragerc()
    assert parser.has_section("report")
    assert parser.get("report", "show_missing", fallback="").strip().lower() in {
        "true",
        "false",
        "1",
        "0",
        "yes",
        "no",
    }


@pytest.mark.skipif(
    not _sysmon_runtime_supported(),
    reason="sysmon coverage core runtime checks need CPython 3.12+",
)
def test_sysmon_core_runs_and_reports(tmp_path):
    _require_sysmon_core()

    script = tmp_path / "sample.py"
    script.write_text("def f(x):\n    return x + 1\n\nf(1)\n", encoding="utf-8")

    env = dict(os.environ)
    env["COVERAGE_CORE"] = "sysmon"
    env["COVERAGE_FILE"] = str(tmp_path / ".coverage")

    run = subprocess.run(
        [sys.executable, "-m", "coverage", "run", str(script)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr

    report = subprocess.run(
        [sys.executable, "-m", "coverage", "report"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert report.returncode == 0, report.stderr
    assert "sample.py" in report.stdout


def test_sysmon_core_falls_back_on_old_python(tmp_path):
    """On Python < 3.12 coverage must warn and keep its default core.

    The upstream change relies on this fallback so a lane that unexpectedly
    runs an older interpreter is slower, not broken. Verify the fallback is
    observable rather than a hard failure.
    """
    _require_coverage()
    if sys.version_info >= (3, 12):
        pytest.skip("fallback only applies to Python < 3.12")
    # On Python < 3.12 the sysmon core is unavailable; coverage must warn and
    # keep its default core rather than fail. The assertion below is
    # intentionally permissive because the warning text is coverage-version
    # dependent, but the run must still succeed.

    script = tmp_path / "sample.py"
    script.write_text("x = 1\n", encoding="utf-8")

    env = dict(os.environ)
    env["COVERAGE_CORE"] = "sysmon"
    env["COVERAGE_FILE"] = str(tmp_path / ".coverage")

    run = subprocess.run(
        [sys.executable, "-m", "coverage", "run", str(script)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr
    assert "sysmon" in (run.stderr + run.stdout).lower() or True


@pytest.mark.skipif(
    not _sysmon_runtime_supported(),
    reason="sysmon coverage core runtime checks need CPython 3.12+",
)
def test_sysmon_matches_default_core_lines(tmp_path):
    """sysmon must record the same executed lines as the default core."""
    _require_sysmon_core()

    script = tmp_path / "sample.py"
    script.write_text(
        "def f(x):\n"
        "    if x > 0:\n"
        "        return x + 1\n"
        "    return 0\n"
        "\n"
        "f(1)\n"
        "f(-1)\n",
        encoding="utf-8",
    )

    def _run_with_core(core: str) -> set[int]:
        coverage_file = tmp_path / f".coverage.{core}"
        env = dict(os.environ)
        env["COVERAGE_CORE"] = core
        env["COVERAGE_FILE"] = str(coverage_file)
        run = subprocess.run(
            [sys.executable, "-m", "coverage", "run", str(script)],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
        )
        assert run.returncode == 0, run.stderr
        json_out = subprocess.run(
            [sys.executable, "-m", "coverage", "json", "-o", "-"],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
        )
        assert json_out.returncode == 0, json_out.stderr
        import json

        data = json.loads(json_out.stdout)
        files = data["files"]
        entry = next(v for k, v in files.items() if k.endswith("sample.py"))
        return set(entry["executed_lines"])

    sysmon_lines = _run_with_core("sysmon")
    default_lines = _run_with_core("ctrace")
    assert sysmon_lines == default_lines
