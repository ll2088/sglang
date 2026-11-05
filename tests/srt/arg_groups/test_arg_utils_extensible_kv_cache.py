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

The switch is intentionally environment-driven today; these tests pin the
contract that the default is off and that an explicit opt-in is visible to
``extensible_kv_cache_enabled`` without touching global state.
"""

import argparse

import pytest

from sglang.srt.arg_groups.arg_utils import A, Arg, add_cli_args_from_dataclass
from sglang.srt.mem_cache.extensible_kv_cache import extensible_kv_cache_enabled


def test_default_is_disabled(monkeypatch):
    monkeypatch.delenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", raising=False)
    assert extensible_kv_cache_enabled() is False


def test_explicit_enable(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    assert extensible_kv_cache_enabled() is True


def test_explicit_disable(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "0")
    assert extensible_kv_cache_enabled() is False


def test_cli_flag_parses_into_dataclass():
    """A dataclass field annotated with Arg is registered and parsed."""

    import dataclasses

    @dataclasses.dataclass
    class _Args:
        enable_extensible_kv_cache: A[
            bool, Arg(help="Enable VMM-backed extensible KV cache.")
        ] = False

    parser = argparse.ArgumentParser()
    add_cli_args_from_dataclass(parser, _Args)

    default = parser.parse_args([])
    assert default.enable_extensible_kv_cache is False

    enabled = parser.parse_args(["--enable-extensible-kv-cache"])
    assert enabled.enable_extensible_kv_cache is True


def test_cli_flag_does_not_mutate_env(monkeypatch):
    """Parsing the flag must not implicitly flip the runtime switch."""

    import dataclasses

    @dataclasses.dataclass
    class _Args:
        enable_extensible_kv_cache: A[
            bool, Arg(help="Enable VMM-backed extensible KV cache.")
        ] = False

    monkeypatch.delenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", raising=False)
    parser = argparse.ArgumentParser()
    add_cli_args_from_dataclass(parser, _Args)
    parser.parse_args(["--enable-extensible-kv-cache"])
    assert extensible_kv_cache_enabled() is False
