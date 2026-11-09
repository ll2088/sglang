import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import verify_run_suite  # noqa: E402


UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.dirname(UNIT_DIR)


def test_exit_codes_are_distinct():
    codes = {
        verify_run_suite.EXIT_OK,
        verify_run_suite.EXIT_TEST_FAILURE,
        verify_run_suite.EXIT_INFRA_FAILURE,
    }
    assert len(codes) == 3
    assert verify_run_suite.EXIT_OK == 0
    assert verify_run_suite.EXIT_INFRA_FAILURE == 2


def test_dimensions_cover_required_names():
    names = [name for name, _ in verify_run_suite.DIMENSIONS]
    assert names == ["collect-diagnostics", "exit-codes", "tests", "usability"]


def test_collect_diagnostics_uses_collect_only():
    args_by_name = dict(verify_run_suite.DIMENSIONS)
    assert "--collect-only" in args_by_name["collect-diagnostics"]
    assert TEST_DIR in args_by_name["collect-diagnostics"]


def test_all_dimensions_disable_cache_provider():
    for name, args in verify_run_suite.DIMENSIONS:
        assert "no:cacheprovider" in args, name


def test_main_returns_zero_when_all_dimensions_pass(monkeypatch, capsys):
    calls = []

    def fake_main(args):
        calls.append(list(args))
        return 0

    monkeypatch.setattr(verify_run_suite.pytest, "main", fake_main)
    code = verify_run_suite.main([])
    out = capsys.readouterr().out
    assert code == verify_run_suite.EXIT_OK
    assert len(calls) == len(verify_run_suite.DIMENSIONS)
    # The actual command and exit code must be printed so CI logs prove pytest
    # really started.
    assert "running: python -m pytest" in out
    assert "exit code: 0" in out


def test_main_returns_one_on_dimension_failure(monkeypatch, capsys):
    def fake_main(args):
        return 1

    monkeypatch.setattr(verify_run_suite.pytest, "main", fake_main)
    code = verify_run_suite.main(["exit-codes"])
    out = capsys.readouterr().out
    assert code == verify_run_suite.EXIT_TEST_FAILURE
    assert "dimension failed: exit-codes" in out


def test_main_selects_single_dimension(monkeypatch):
    calls = []

    def fake_main(args):
        calls.append(list(args))
        return 0

    monkeypatch.setattr(verify_run_suite.pytest, "main", fake_main)
    code = verify_run_suite.main(["usability"])
    assert code == verify_run_suite.EXIT_OK
    assert len(calls) == 1
    assert any("test_run_suite_unit_config.py" in a for a in calls[0])


def test_main_rejects_unknown_dimension(monkeypatch, capsys):
    monkeypatch.setattr(verify_run_suite.pytest, "main", lambda args: 0)
    code = verify_run_suite.main(["not-a-dimension"])
    err = capsys.readouterr().err
    assert code == verify_run_suite.EXIT_INFRA_FAILURE
    assert "unknown dimension" in err


def test_main_returns_infra_code_without_pytest(monkeypatch, capsys):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pytest":
            raise ImportError("pytest intentionally unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    code = verify_run_suite.main([])
    err = capsys.readouterr().err
    assert code == verify_run_suite.EXIT_INFRA_FAILURE
    assert "test/unit/requirements.txt" in err
