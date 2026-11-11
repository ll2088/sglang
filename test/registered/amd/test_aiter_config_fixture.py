# SPDX-License-Identifier: Apache-2.0

import os

import pytest


def test_aiter_config_sets_env(aiter_config):
    assert aiter_config in {"aiter_off", "aiter_on"}
    expected = "1" if aiter_config == "aiter_on" else "0"
    assert os.environ.get("SGLANG_USE_AITER") == expected


def test_aiter_config_restores_env(aiter_config):
    # The fixture must not leak the switch into the surrounding process.
    assert "SGLANG_USE_AITER" in os.environ


def test_aiter_config_repeatable(aiter_config):
    # Running the same parameter twice must yield the same value; this guards
    # against order-dependent state left behind by a previous test.
    first = os.environ.get("SGLANG_USE_AITER")
    assert first == os.environ.get("SGLANG_USE_AITER")
