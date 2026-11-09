import argparse
import configparser
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import run_suite  # noqa: E402


class _FakeTest:
    def __init__(self, filename, backend, suite, nightly=False, disabled=None):
        self.filename = filename
        self.backend = backend
        self.effective_suite = suite
        self.nightly = nightly
        self.disabled = disabled
        self.est_time = 1.0


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


def _patch_collect(monkeypatch, fn):
    monkeypatch.setattr(run_suite, "collect_tests", fn)


def test_collect_failure_returns_dedicated_code(monkeypatch, capsys):
    def boom(files, sanity_check=True):
        raise RuntimeError("bad registration in test_x.py")

    _patch_collect(monkeypatch, boom)
    code = run_suite.run_a_suite(_make_args())
    out = capsys.readouterr().out
    assert code == run_suite.EXIT_COLLECT_FAILURE
    assert "COLLECTION FAILED" in out
    assert "bad registration in test_x.py" in out
    assert "RuntimeError" in out


def test_execution_failure_code_differs_from_collect(monkeypatch):
    _patch_collect(
        monkeypatch,
        lambda files, sanity_check=True: [
            _FakeTest("/tmp/test_a.py", run_suite.HWBackend.CPU, "base-a-test-cpu")
        ],
    )
    monkeypatch.setattr(run_suite, "run_unittest_files", lambda *a, **k: 1)
    code = run_suite.run_a_suite(_make_args())
    assert code == run_suite.EXIT_TEST_FAILURE
    assert code != run_suite.EXIT_COLLECT_FAILURE


def test_no_tests_returns_dedicated_code(monkeypatch, capsys):
    _patch_collect(monkeypatch, lambda files, sanity_check=True: [])
    code = run_suite.run_a_suite(_make_args())
    out = capsys.readouterr().out
    assert code == run_suite.EXIT_NO_TESTS
    assert "No tests to run" in out


def test_all_pass_returns_zero(monkeypatch):
    _patch_collect(
        monkeypatch,
        lambda files, sanity_check=True: [
            _FakeTest("/tmp/test_a.py", run_suite.HWBackend.CPU, "base-a-test-cpu")
        ],
    )
    monkeypatch.setattr(run_suite, "run_unittest_files", lambda *a, **k: 0)
    code = run_suite.run_a_suite(_make_args())
    assert code == run_suite.EXIT_OK


def test_exit_codes_are_pairwise_distinct():
    codes = {
        run_suite.EXIT_OK,
        run_suite.EXIT_TEST_FAILURE,
        run_suite.EXIT_COLLECT_FAILURE,
        run_suite.EXIT_NO_TESTS,
    }
    assert len(codes) == 4
    assert run_suite.EXIT_OK == 0


def test_pytest_ini_is_loadable():
    ini_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pytest.ini")
    assert os.path.exists(ini_path)
    parser = configparser.ConfigParser()
    parser.read(ini_path)
    assert parser.has_section("pytest")
    assert parser.get("pytest", "asyncio_mode") == "auto"
