# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""CLI parsing for the opt-in extensible KV cache switch.

The switch is read from the environment by ``extensible_kv_cache_enabled``;
this test pins the default (off) and the explicit opt-in/opt-out behaviour so
that the fallback path stays the default.
"""

import argparse
import os

import pytest

from sglang.srt.mem_cache.extensible_kv_cache import (
    extensible_kv_cache_enabled,
)


def _parse(env_value):
    """Parse the switch the way the server would, without importing the
    full ServerArgs (which pulls in the model stack)."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--enable-extensible-kv-cache",
        action="store_true",
        default=env_value == "1",
    )
    return parser.parse_args([])


def test_default_is_disabled(monkeypatch):
    monkeypatch.delenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", raising=False)
    assert extensible_kv_cache_enabled() is False
    assert _parse(None).enable_extensible_kv_cache is False


def test_explicit_enable(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    assert extensible_kv_cache_enabled() is True
    assert _parse("1").enable_extensible_kv_cache is True


def test_explicit_disable(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "0")
    assert extensible_kv_cache_enabled() is False
    assert _parse("0").enable_extensible_kv_cache is False


@pytest.mark.parametrize("value", ["", "true", "yes", "on", "2"])
def test_only_exact_one_enables(monkeypatch, value):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", value)
    assert extensible_kv_cache_enabled() is False
