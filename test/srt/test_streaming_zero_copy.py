"""Zero-copy streaming token history helpers (standard library only)."""

import importlib.util
import pathlib

import pytest


def _load_managers_utils():
    path = (
        pathlib.Path(__file__).resolve().parents[2]
        / "python"
        / "sglang"
        / "srt"
        / "managers"
        / "utils.py"
    )
    spec = importlib.util.spec_from_file_location("sglang_managers_utils", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
 def managers_utils():
    return _load_managers_utils()


def test_view_matches_full_copy(managers_utils):
    history = [10, 20, 21]
    sent = 1
    view = managers_utils.token_history_view(history, sent)
    assert list(view) == list(history)[sent:]
    assert managers_utils.token_history_len(history) == 3


def test_view_does_not_copy_or_mutate(managers_utils):
    history = [1, 2, 3]
    view = managers_utils.token_history_view(history, 0)
    assert view is history
    history.append(4)
    assert list(view) == [1, 2, 3, 4]


def test_empty_history(managers_utils):
    assert managers_utils.token_history_view(None, 0) == ()
    assert managers_utils.token_history_view([], 0) == []
    assert managers_utils.token_history_len(None) == 0
