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

"""CLI parsing tests for the extensible KV cache opt-in switch.

The switch is exposed through the environment variable
``SGLANG_ENABLE_EXTENSIBLE_KV_CACHE``; these tests pin the default (off) and
the explicit opt-in/opt-out behaviour so the fallback path stays the default.
"""

import pytest

from sglang.srt.mem_cache.extensible_kv_cache import extensible_kv_cache_enabled


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, False),
        ("0", False),
        ("1", True),
        ("", False),
        ("true", False),
    ],
)
def test_env_switch_parsing(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", raising=False)
    else:
        monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", value)
    assert extensible_kv_cache_enabled() is expected


def test_explicit_disable_keeps_fallback(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "0")
    assert extensible_kv_cache_enabled() is False


def test_explicit_enable(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    assert extensible_kv_cache_enabled() is True
