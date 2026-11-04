# Copyright (c) 2025, Tri Dao.
"""Pure-Python rule for dense 2CTA MLA Q-head padding.

This module deliberately has no CUDA/CuTe/sglang imports so it can be loaded
by file path on CPU-only runners and by pack_gqa as a fallback when the
package config module is unavailable.
"""

__all__ = [
    "mla_2cta_pads_dense_qheads",
    "mla_2cta_accepts_dense_qheads",
]


def mla_2cta_pads_dense_qheads(nheads_per_kv: int, num_heads_kv) -> bool:
    """Whether dense 2CTA MLA pads Q heads to its 128-row tile.

    The rule is MQA (a single KV head) with 65 to 127 Q heads per KV head.
    ``num_heads_kv`` may be an int count or a boolean MQA flag; both forms are
    accepted so the kernel and the config validation can share one predicate.
    """
    if isinstance(num_heads_kv, bool):
        is_mqa = num_heads_kv
    else:
        is_mqa = num_heads_kv == 1
    return is_mqa and 64 < nheads_per_kv < 128


def mla_2cta_accepts_dense_qheads(nheads_per_kv: int, num_heads_kv) -> bool:
    """Whether dense 2CTA MLA accepts this Q-head count.

    Heads that divide the 128-row tile are accepted as-is; MQA with 65 to 127
    heads is accepted by padding to the tile. Everything else is rejected.
    """
    if nheads_per_kv <= 0 or nheads_per_kv > 128:
        return False
    if 128 % nheads_per_kv == 0:
        return True
    return mla_2cta_pads_dense_qheads(nheads_per_kv, num_heads_kv)
