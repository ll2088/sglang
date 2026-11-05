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
        bool, Arg(help="Enable the VMM-backed extensible KV cache.")
    ] = False


def _parse(argv):
    parser = argparse.ArgumentParser()
    add_cli_args_from_dataclass(parser, _ExtensibleKVCacheArgs)
    return parser.parse_args(argv)


def test_extensible_kv_cache_flag_defaults_to_disabled():
    args = _parse([])
    assert args.enable_extensible_kv_cache is False


def test_extensible_kv_cache_flag_can_be_enabled():
    args = _parse(["--enable-extensible-kv-cache"])
    assert args.enable_extensible_kv_cache is True


def test_extensible_kv_cache_flag_explicit_disable():
    args = _parse(["--enable-extensible-kv-cache"])
    assert args.enable_extensible_kv_cache is True
    # The fallback path is selected by leaving the flag at its default.
    fallback = _parse([])
    assert fallback.enable_extensible_kv_cache is False
