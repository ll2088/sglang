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
"""Argument parsing tests for the extensible KV cache switch.

The switch is currently an environment variable, so these tests pin the
contract the CLI/arg layer relies on: the default is off, an explicit value
enables it, and the fallback path is selected when the feature is off.
"""

import argparse

from sglang.srt.arg_groups.arg_utils import (
    A,
    Arg,
    add_cli_args_from_dataclass,
    _env_flag_default,
)
from sglang.srt.mem_cache.extensible_kv_cache import extensible_kv_cache_enabled

import dataclasses


@dataclasses.dataclass
class _Args:
    enable_extensible_kv_cache: A[
        bool, Arg(help="Enable the VMM-backed extensible KV cache.")
    ] = False


def test_cli_flag_defaults_off():
    parser = argparse.ArgumentParser()
    add_cli_args_from_dataclass(parser, _Args)
    args = parser.parse_args([])
    assert args.enable_extensible_kv_cache is False


def test_cli_flag_can_be_enabled():
    parser = argparse.ArgumentParser()
    add_cli_args_from_dataclass(parser, _Args)
    args = parser.parse_args(["--enable-extensible-kv-cache"])
    assert args.enable_extensible_kv_cache is True


def test_env_flag_default_helper(monkeypatch):
    monkeypatch.delenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", raising=False)
    assert _env_flag_default("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE") is False
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    assert _env_flag_default("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE") is True
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "off")
    assert _env_flag_default("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE") is False


def test_runtime_switch_matches_env(monkeypatch):
    monkeypatch.delenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", raising=False)
    assert extensible_kv_cache_enabled() is False
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    assert extensible_kv_cache_enabled() is True
