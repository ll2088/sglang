"""Sanity checks for the repository coverage configuration.

These tests are intentionally lightweight: they do not run the full suite under
coverage, they only verify that the configuration is well-formed and that the
``sysmon`` coverage core can start and produce a report when requested.
"""

import configparser
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
COVERAGERC = REPO_ROOT / ".coveragerc"


def _coverage_available() -> bool:
    return importlib.util.find_spec("coverage") is not None


def _require_coverage() -> None:
    """Skip the runtime checks when coverage is not installed.

    The configuration assertions above still run everywhere; only the
    subprocess-based sysmon checks need the coverage package. Skipping with a
    clear reason keeps environment problems from being reported as code
    failures.
    """
    if not _coverage_available():
        pytest.skip("coverage is not installed; skipping sysmon runtime checks")


def _require_sysmon_core() -> None:
    """Skip when the installed coverage cannot provide the sysmon core.

    ``core = sysmon`` needs Python 3.12+ and a coverage release that ships the
    ``sys.monitoring`` core. Older environments fall back to the default core
    with a warning, so the runtime assertions below would be meaningless there.
    """
    _require_coverage()
    if sys.version_info < (3, 12):
        pytest.skip("sysmon coverage core requires Python 3.12+")
    try:
        from coverage.core import Core  # noqa: F401
    except Exception:  # pragma: no cover - very old coverage layouts
        pass
    try:
        from coverage import coverage as _coverage_cls

        cov = _coverage_cls(core="sysmon")
        cov.start()
        cov.stop()
    except Exception as exc:  # pragma: no cover - depends on environment
        pytest.skip(f"installed coverage cannot start the sysmon core: {exc}")


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
    sys.version_info < (3, 12),
    reason="sysmon coverage core requires Python 3.12+",
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


@pytest.mark.skipif(
    sys.version_info < (3, 12),
    reason="sysmon coverage core requires Python 3.12+",
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
