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
"""Argument parsing tests for the opt-in extensible KV cache switch.

These tests exercise the environment-variable switch that gates the VMM-backed
KV cache path. They do not require CUDA: the switch is read from the process
environment and must default to disabled so that the existing estimate-based
allocation path remains the default.
"""

import pytest

from sglang.srt.mem_cache.extensible_kv_cache import (
    extensible_kv_cache_enabled,
    plan_extensible_kv_cache,
    ExtensibleKVCacheConfig,
)


def test_switch_defaults_to_disabled(monkeypatch):
    monkeypatch.delenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", raising=False)
    assert extensible_kv_cache_enabled() is False


def test_switch_enabled_by_env(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    assert extensible_kv_cache_enabled() is True


def test_switch_explicit_disable_falls_back(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "0")
    assert extensible_kv_cache_enabled() is False
    config = ExtensibleKVCacheConfig(
        bytes_per_block=1 << 20,
        max_num_blocks=4,
        device_index=0,
    )
    # Disabled switch must short-circuit before touching the driver.
    assert plan_extensible_kv_cache(config) is None


@pytest.mark.parametrize("value", ["", "false", "no", "off", "2"])
def test_switch_rejects_non_enabled_values(monkeypatch, value):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", value)
    assert extensible_kv_cache_enabled() is False
