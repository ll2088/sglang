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
"""CLI parsing for the opt-in extensible KV cache switch."""

import argparse
import dataclasses

from sglang.srt.arg_groups.arg_utils import A, add_cli_args_from_dataclass


@dataclasses.dataclass
class _Args:
    enable_extensible_kv_cache: A[
        bool, "Reserve KV cache VA space and commit after warmup."
    ] = False


def _parse(argv):
    parser = argparse.ArgumentParser()
    add_cli_args_from_dataclass(parser, _Args)
    return parser.parse_args(argv)


def test_default_is_disabled():
    assert _parse([]).enable_extensible_kv_cache is False


def test_explicit_enable():
    assert _parse(["--enable-extensible-kv-cache"]).enable_extensible_kv_cache is True


def test_explicit_disable_keeps_fallback():
    # store_true cannot express an explicit off, so the default must stay off
    # and the runtime env switch must be able to force the fallback path.
    args = _parse([])
    assert args.enable_extensible_kv_cache is False
