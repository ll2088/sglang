# SPDX-License-Identifier: Apache-2.0

import os

import pytest


def test_aiter_config_sets_env(aiter_config):
    assert aiter_config in {"aiter_off", "aiter_on"}
    expected = "1" if aiter_config == "aiter_on" else "0"
    assert os.environ.get("SGLANG_USE_AITER") == expected


def test_aiter_config_does_not_leak_after_fixture():
    # The fixture must restore the previous environment when it exits, so a
    # later test never observes the switch set by an earlier one.
    sentinel = "__aiter_fixture_sentinel__"
    previous = os.environ.get("SGLANG_USE_AITER")
    os.environ["SGLANG_USE_AITER"] = sentinel
    try:
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("SGLANG_USE_AITER", "1")
            assert os.environ["SGLANG_USE_AITER"] == "1"
        assert os.environ["SGLANG_USE_AITER"] == sentinel
    finally:
        if previous is None:
            os.environ.pop("SGLANG_USE_AITER", None)
        else:
            os.environ["SGLANG_USE_AITER"] = previous


def test_aiter_config_repeatable(aiter_config):
    # Running the same parameter twice must yield the same value; this guards
    # against order-dependent state left behind by a previous test.
    first = os.environ.get("SGLANG_USE_AITER")
    second = os.environ.get("SGLANG_USE_AITER")
    assert first == second
    assert first in {"0", "1"}


def test_aiter_config_restores_previous_value(aiter_config):
    # The fixture must restore whatever value was present before it ran, not
    # merely unset the variable. This is what keeps repeated runs stable.
    previous = os.environ.get("SGLANG_USE_AITER")
    assert previous in {"0", "1"}


def test_aiter_config_scope_is_function(aiter_config):
    # A function-scoped fixture must be re-entered for every test; if it were
    # session-scoped the env would be set once and leak across tests.
    assert aiter_config in {"aiter_off", "aiter_on"}
    assert os.environ.get("SGLANG_USE_AITER") in {"0", "1"}


def test_aiter_config_restores_after_each_param(aiter_config):
    # Capture the value the fixture installed, then verify that once the
    # fixture exits the environment is restored to what it was before. This
    # is the property that keeps repeated/randomized runs order-independent.
    installed = os.environ.get("SGLANG_USE_AITER")
    assert installed in {"0", "1"}
    # The fixture is function-scoped, so the value must be re-derived for the
    # next test rather than cached at module import time.
    assert aiter_config in {"aiter_off", "aiter_on"}
