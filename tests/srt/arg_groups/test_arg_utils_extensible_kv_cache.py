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
this test pins the default (off) and the explicit opt-in/opt-out behaviour so
the fallback path stays reachable from the command line.
"""

from __future__ import annotations

import argparse
import dataclasses
from typing import Annotated

import pytest

from sglang.srt.arg_groups.arg_utils import Arg, add_cli_args_from_dataclass
from sglang.srt.mem_cache.extensible_kv_cache import extensible_kv_cache_enabled


@dataclasses.dataclass
class _Args:
    enable_extensible_kv_cache: Annotated[
        bool, Arg(help="Enable the VMM-backed extensible KV cache.")
    ] = False


def _parse(argv):
    parser = argparse.ArgumentParser()
    add_cli_args_from_dataclass(parser, _Args)
    return parser.parse_args(argv)


def test_default_is_off():
    args = _parse([])
    assert args.enable_extensible_kv_cache is False


def test_explicit_enable():
    args = _parse(["--enable-extensible-kv-cache"])
    assert args.enable_extensible_kv_cache is True


def test_env_default_off(monkeypatch):
    monkeypatch.delenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", raising=False)
    assert not extensible_kv_cache_enabled()


def test_env_explicit_disable(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "0")
    assert not extensible_kv_cache_enabled()


def test_env_explicit_enable(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", "1")
    assert extensible_kv_cache_enabled()


def test_cli_flag_does_not_change_env(monkeypatch):
    monkeypatch.delenv("SGLANG_ENABLE_EXTENSIBLE_KV_CACHE", raising=False)
    args = _parse(["--enable-extensible-kv-cache"])
    assert args.enable_extensible_kv_cache is True
    # The runtime check is env-driven; parsing alone must not flip it.
    assert not extensible_kv_cache_enabled()
