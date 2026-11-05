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
"""CLI parsing tests for the extensible KV cache switch.

The switch is read from the environment by ``extensible_kv_cache_enabled``;
this test pins the default (off) and the explicit opt-in so a future change to
the default is caught.
"""

from __future__ import annotations

import pytest

from sglang.srt.mem_cache import extensible_kv_cache as ekc


def test_default_is_disabled(monkeypatch):
    monkeypatch.delenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", raising=False)
    assert ekc.extensible_kv_cache_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "yes", "on"])
def test_explicit_enable(monkeypatch, value):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", value)
    # Only the documented "1" enables the path; other spellings stay off so
    # the switch is unambiguous across shells and CI configs.
    assert ekc.extensible_kv_cache_enabled() is (value == "1")


def test_explicit_disable(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "0")
    assert ekc.extensible_kv_cache_enabled() is False
