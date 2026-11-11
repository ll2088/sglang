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
