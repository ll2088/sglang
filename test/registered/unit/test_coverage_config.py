"""Sanity checks for the repository coverage configuration.

These tests are intentionally lightweight: they do not run the full suite under
coverage, they only verify that the configuration is well-formed and that the
``sysmon`` coverage core can start and produce a report when requested.
"""

import configparser
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
COVERAGERC = REPO_ROOT / ".coveragerc"


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
    pytest.importorskip("coverage")

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
