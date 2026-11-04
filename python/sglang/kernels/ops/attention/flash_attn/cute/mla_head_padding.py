# Copyright (c) 2025, Tri Dao.
"""Pure-Python MLA head-padding rules.

This module intentionally has no CUDA/CuTe imports so the routing predicate can
be unit-tested on any runner (and imported by config validation) without
pulling in the kernel dependencies.
"""


def mla_2cta_pads_dense_qheads(nheads_per_kv: int, num_heads_kv) -> bool:
    """Whether dense 2CTA MLA pads Q heads to its 128-row tile: MQA with 65 to 127 heads.

    ``num_heads_kv`` is the number of KV heads; a truthy value of 1 (or ``True``)
    denotes MQA. Any other count, including 0, disables the padding rule.

    This is the single source of truth shared by the 2CTA kernel constructor
    (``flash_fwd_mla_sm100``) and ``validate_mla_fwd_config``: dense MQA with 65
    to 127 Q heads per KV head is padded in-kernel to 128 rows, while fewer heads
    keep raising because padding would waste over half of each tile's MMA rows.
    """
    return bool(num_heads_kv) and num_heads_kv == 1 and 64 < nheads_per_kv < 128
