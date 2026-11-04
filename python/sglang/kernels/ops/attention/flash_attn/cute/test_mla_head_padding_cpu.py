# Copyright (c) 2025, Tri Dao.
"""CPU-only smoke tests for the dense 2CTA MLA head-padding rule.

This module is a thin alias entry point so CI can select the CPU-only tests by
file name without installing sglang or CUDA deps. The actual assertions live in
test_mla_head_padding_standalone.py; this file re-exports them so both names
work in test selection.
"""

import importlib.util
import pathlib

_STANDALONE = pathlib.Path(__file__).with_name("test_mla_head_padding_standalone.py")
_spec = importlib.util.spec_from_file_location(
    "_fa_cute_mla_head_padding_standalone", _STANDALONE
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(module)

mla_2cta_pads_dense_qheads = module.mla_2cta_pads_dense_qheads

test_mla_2cta_pads_dense_qheads = module.test_mla_2cta_pads_dense_qheads
test_padded_heads_map_to_identity_head_order = (
    module.test_padded_heads_map_to_identity_head_order
)
test_non_padded_heads_keep_identity_order = (
    module.test_non_padded_heads_keep_identity_order
)
test_accepts_dense_qheads_matches_padding_rule = (
    module.test_accepts_dense_qheads_matches_padding_rule
)
test_predicate_resolves_without_config_module = (
    module.test_predicate_resolves_without_config_module
)
test_predicate_resolves_without_sglang_package = (
    module.test_predicate_resolves_without_sglang_package
)
test_pack_gqa_predicate_fallback_without_config_module = (
    module.test_pack_gqa_predicate_fallback_without_config_module
)
test_predicate_resolves_in_isolated_subprocess = (
    module.test_predicate_resolves_in_isolated_subprocess
)
