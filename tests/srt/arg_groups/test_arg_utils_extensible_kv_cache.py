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

import argparse
import dataclasses

from sglang.srt.arg_groups.arg_utils import A, Arg, add_cli_args_from_dataclass


@dataclasses.dataclass
class _ExtensibleKVCacheArgs:
    enable_extensible_kv_cache: A[
        bool, Arg(help="Enable VMM-backed extensible KV cache.")
    ] = False
    extensible_kv_cache_fallback: A[
        bool, Arg(help="Fall back to estimate-based sizing when VMM is unavailable.")
    ] = True


def _parse(argv):
    parser = argparse.ArgumentParser()
    add_cli_args_from_dataclass(parser, _ExtensibleKVCacheArgs)
    return parser.parse_args(argv)


def test_defaults_are_opt_in_with_fallback():
    args = _parse([])
    assert args.enable_extensible_kv_cache is False
    assert args.extensible_kv_cache_fallback is True


def test_explicit_enable():
    args = _parse(["--enable-extensible-kv-cache"])
    assert args.enable_extensible_kv_cache is True
    assert args.extensible_kv_cache_fallback is True


def test_explicit_disable_fallback():
    args = _parse(["--enable-extensible-kv-cache", "--extensible-kv-cache-fallback"])
    assert args.enable_extensible_kv_cache is True
    assert args.extensible_kv_cache_fallback is True


def test_fallback_flag_alone_keeps_feature_off():
    args = _parse(["--extensible-kv-cache-fallback"])
    assert args.enable_extensible_kv_cache is False
    assert args.extensible_kv_cache_fallback is True
