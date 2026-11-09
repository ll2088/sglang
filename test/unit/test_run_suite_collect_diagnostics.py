import argparse
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# conftest.py installs the sglang.test.ci stubs before collection, so
# run_suite imports cleanly even when the full sglang runtime is absent.
import run_suite  # noqa: E402


def _make_args(**overrides):
    defaults = dict(
        hw="cpu",
        suite="base-a-test-cpu",
        nightly=False,
        timeout_per_file=10,
        timeout_override=[],
        timeout_from_est_time=False,
        continue_on_error=False,
        auto_partition_id=None,
        auto_partition_size=None,
        enable_retry=False,
        max_attempts=2,
        retry_wait_seconds=0,
        retry_timeout_increase=0,
        partition_model_file=None,
        only_file_list=None,
        fork_worker_batch_size=1,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_collect_failure_reports_file_and_reason(monkeypatch, capsys):
    """A collection exception must surface the failing file and reason."""

    def boom(files, sanity_check=True):
        raise ValueError("invalid suite for test/registered/core/test_x.py")

    monkeypatch.setattr(run_suite, "collect_tests", boom)
    code = run_suite.run_a_suite(_make_args())
    out = capsys.readouterr().out
    assert code == run_suite.EXIT_COLLECT_FAILURE
    assert "COLLECTION FAILED" in out
    assert "ValueError" in out
    assert "invalid suite for test/registered/core/test_x.py" in out
    assert "Traceback" in out
    # Greppable status marker must be present so CI logs can be checked
    # mechanically and collection failures are not confused with exec failures.
    assert "collect-failed" in out


def test_validate_all_suites_failure_is_collect_failure(monkeypatch, capsys):
    """A bad registration caught by validate_all_suites is a collect failure."""

    class _BadTest:
        filename = "/tmp/test_bad.py"
        backend = run_suite.HWBackend.CPU
        effective_suite = "not-a-real-suite"
        nightly = False
        disabled = None
        est_time = 1.0

    monkeypatch.setattr(
        run_suite, "collect_tests", lambda files, sanity_check=True: [_BadTest()]
    )
    code = run_suite.run_a_suite(_make_args())
    out = capsys.readouterr().out
    assert code == run_suite.EXIT_COLLECT_FAILURE
    assert "COLLECTION FAILED" in out
    assert "not-a-real-suite" in out


def test_collect_failure_names_failing_file_and_reason(monkeypatch, capsys):
    """The diagnostic must name the failing file path and the error summary."""

    def boom(files, sanity_check=True):
        raise RuntimeError(
            "cannot import test/registered/core/test_broken.py: NameError"
        )

    monkeypatch.setattr(run_suite, "collect_tests", boom)
    code = run_suite.run_a_suite(_make_args())
    out = capsys.readouterr().out
    assert code == run_suite.EXIT_COLLECT_FAILURE
    assert "COLLECTION FAILED" in out
    assert "test/registered/core/test_broken.py" in out
    assert "NameError" in out
    assert "RuntimeError" in out


def test_collect_failure_does_not_run_tests(monkeypatch):
    """Collection failure must short-circuit before executing any test."""
    called = {"run": False}

    def boom(files, sanity_check=True):
        raise RuntimeError("bad registration")

    def fake_run(*a, **k):
        called["run"] = True
        return 0

    monkeypatch.setattr(run_suite, "collect_tests", boom)
    monkeypatch.setattr(run_suite, "run_unittest_files", fake_run)
    code = run_suite.run_a_suite(_make_args())
    assert code == run_suite.EXIT_COLLECT_FAILURE
    assert called["run"] is False
