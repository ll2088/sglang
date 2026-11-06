"""Host-only forward config selection and validation.

FwdConfig uses exact split counts: 1 disables SplitKV. Explicit configs are
validated, not normalized. Selection must not read tensor contents, synchronize
CUDA, or allocate device memory. Nothing routes through this module yet; that is
a separate change so the routing diff stays about plumbing and the tuning diffs
stay about policy.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import NamedTuple, Optional, Union


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


@dataclass(frozen=True)
class Sm90Registers:
    """Per-thread setmaxnreg targets: each MMA warp group raises, the producer lowers."""

    mma: int
    producer: int


@dataclass(frozen=True)
class Sm100Registers:
    """Per-thread setmaxnreg targets for the SM100/SM110 forward warp-group roles.

    One softmax warp group per Q stage raises to ``softmax``; the remaining
    ``3 - q_stage`` load/MMA/epilogue groups lower to ``other``; correction raises or
    lowers depending on whether ``correction`` exceeds the launch register count.
    """

    softmax: int
    correction: int
    other: int


# None for architectures whose forward kernels do not split registers by role.
Registers = Union[Sm90Registers, Sm100Registers, None]


@dataclass(frozen=True)
class FwdConfig:
    """Fully resolved forward launch configuration.

    Dynamic persistence is derived from scheduler metadata, not is_static_persistent.
    num_splits and tile_n affect reduction order. With causal/local SplitKV, the
    per-split KV range also depends on the M tile extent, q_stage * tile_m.
    """

    device_capacity: int
    tile_m: int
    tile_n: int
    mma_pv_is_rs: bool
    intra_wg_overlap: bool
    q_stage: int
    use_clc_scheduler: bool
    is_static_persistent: bool
    use_tma_o: bool
    use_s_ping_pong: bool
    num_splits: int
    use_2cta_instrs: bool
    registers: Registers


class FwdHeuristicInputs(NamedTuple):
    """Host metadata for config selection and validation.

    has_max_seqlen_q_hint distinguishes a caller's varlen Q bound from total_q.
    """

    device_arch: int
    num_sms: int
    dtype: str
    head_dim: int
    head_dim_v: int
    num_heads: int
    num_heads_kv: int
    batch_size: int
    total_q: int
    total_k: int
    max_seqlen_q: int
    max_seqlen_k: int
    has_max_seqlen_q_hint: bool = True
    seqlen_k_per_split: Optional[int] = None
    causal: bool = False
    local: bool = False
    window_size_left: Optional[int] = None
    window_size_right: Optional[int] = None
    has_cu_seqlens_q: bool = False
    has_cu_seqlens_k: bool = False
    has_seqused_q: bool = False
    has_seqused_k: bool = False
    has_caller_scheduler_metadata: bool = False
    has_tile_count_semaphore: bool = False
    pack_gqa: bool = False
    page_size: Optional[int] = None
    use_block_sparsity: bool = False
    sparse_q_block_size: Optional[int] = None
    sparse_kv_block_size: Optional[int] = None
    has_score_mod: bool = False
    has_mask_mod: bool = False
    has_learnable_sink: bool = False
    has_lse: bool = False
    requested_tile_m: Optional[int] = None
    requested_tile_n: Optional[int] = None
    requested_mma_pv_is_rs: Optional[bool] = None
    requested_intra_wg_overlap: Optional[bool] = None
    requested_num_splits: Optional[int] = None
    requested_use_clc_scheduler: bool = False
    disable_2cta: bool = False
    disable_s_ping_pong: bool = False


# ---------------------------------------------------------------------------
# Tuning tables (moved out of the kernels' former inline tables).
# ---------------------------------------------------------------------------

# (q_stage, softmax, correction, other) keyed by (head_dim, head_dim_v).
SM100_TUNING = {
    (64, 64): (2, 192, 128, 96),
    (128, 128): (2, 208, 128, 96),
    (192, 128): (2, 224, 128, 96),
    (256, 256): (1, 256, 128, 80),
}

# FP8 uses one extra Q stage when the KV blocks allow it.
SM100_FP8_TUNING = {
    (64, 64): (3, 232, 128, 80),
    (128, 128): (3, 240, 128, 80),
    (192, 128): (3, 232, 128, 80),
    (256, 256): (2, 240, 128, 72),
}

# SM90 TMA-Q/KV register rule: (mma, producer).
SM90_REGISTERS = Sm90Registers(mma=224, producer=48)

SM100_MAX_REGISTER_BUDGET = 512
SM90_MAX_REGISTER_BUDGET = 512
SM100_SOFTMAX_MIN = 128
SM100_OTHER_MAX = 128


def _is_fp8(dtype: str) -> bool:
    return "float8" in dtype or "e4m3" in dtype or "e5m2" in dtype
+
+
+def _is_sm100_family(device_arch: int) -> bool:
+    return device_arch in (100, 103, 110)
+
+
+def _sm100_tuning(inputs: FwdHeuristicInputs):
+    key = (inputs.head_dim, inputs.head_dim_v)
+    if _is_fp8(inputs.dtype) and key in SM100_FP8_TUNING:
+        return SM100_FP8_TUNING[key]
+    return SM100_TUNING.get(key)
+
+
+# ---------------------------------------------------------------------------
+# Shared support predicates: selection and validation both call these.
+# ---------------------------------------------------------------------------
+
+
+def is_2cta_eligible(inputs: FwdHeuristicInputs) -> bool:
+    """2CTA needs an even number of packed Q tiles and hd in {128, 192}."""
+    if inputs.disable_2cta or not _is_sm100_family(inputs.device_arch):
+        return False
+    if inputs.head_dim not in (128, 192) or inputs.head_dim_v != inputs.head_dim:
+        return False
+    qhead_per_kvhead = (
+        inputs.num_heads // inputs.num_heads_kv if inputs.pack_gqa else 1
+    )
+    tile_m = 128
+    num_m_tiles = ceil_div(inputs.max_seqlen_q * qhead_per_kvhead, tile_m)
+    return num_m_tiles % 2 == 0
+
+
+def can_use_clc(inputs: FwdHeuristicInputs) -> bool:
+    """CLC needs cluster launch and no tile-count semaphore."""
+    if not _is_sm100_family(inputs.device_arch):
+        return False
+    if inputs.has_tile_count_semaphore:
+        return False
+    return inputs.has_caller_scheduler_metadata
+
+
+def can_use_static_persistent(inputs: FwdHeuristicInputs) -> bool:
+    """Static persistence needs a known tile count and a single batch."""
+    if not _is_sm100_family(inputs.device_arch):
+        return False
+    if inputs.has_cu_seqlens_q or inputs.has_tile_count_semaphore:
+        return False
+    return True
+
+
+def can_use_tma_o(inputs: FwdHeuristicInputs) -> bool:
+    """TMA store to O requires a dense, non-varlen Q layout."""
+    if not _is_sm100_family(inputs.device_arch):
+        return False
+    if inputs.has_cu_seqlens_q or inputs.has_seqused_q:
+        return False
+    if inputs.has_score_mod or inputs.has_mask_mod:
+        return False
+    return True
+
+
+def can_use_s_ping_pong(inputs: FwdHeuristicInputs) -> bool:
+    """S ping-pong needs at least 2 KV stages worth of work per M tile."""
+    if not _is_sm100_family(inputs.device_arch):
+        return False
+    if inputs.disable_s_ping_pong:
+        return False
+    if inputs.use_block_sparsity:
+        return False
+    return ceil_div(inputs.max_seqlen_k, 64) >= 2
+
+
+# ---------------------------------------------------------------------------
+# Register selection / validation.
+# ---------------------------------------------------------------------------
+
+
+def select_registers(
+    inputs: FwdHeuristicInputs,
+    tile_m: int,
+    tile_n: int,
+    q_stage: int,
+    use_2cta_instrs: bool,
+) -> Registers:
+    """Resolve the per-role register split after the final tile/stage/2CTA choices."""
+    if _is_sm100_family(inputs.device_arch):
+        tuning = _sm100_tuning(inputs)
+        if tuning is None:
+            softmax = 232
+            correction = 128
+            other = 96
+        else:
+            tuning_q_stage, softmax, correction, other = tuning
+            if not _is_fp8(inputs.dtype):
+                softmax = max(softmax, SM100_SOFTMAX_MIN)
+        return Sm100Registers(softmax=softmax, correction=correction, other=other)
+    if inputs.device_arch == 90:
+        return SM90_REGISTERS
+    return None
+
+
+def _register_direction_errors(registers: Registers, device_arch: int):
+    errors = []
+    if isinstance(registers, Sm100Registers):
+        if registers.softmax < SM100_SOFTMAX_MIN:
+            errors.append(
+                f"SM100 softmax registers must be >= {SM100_SOFTMAX_MIN}, "
+                f"got {registers.softmax}"
+            )
+        if registers.other > SM100_OTHER_MAX:
+            errors.append(
+                f"SM100 other registers must be <= {SM100_OTHER_MAX}, "
+                f"got {registers.other}"
+            )
+    elif isinstance(registers, Sm90Registers):
+        if not registers.producer <= registers.mma:
+            errors.append(
+                "SM90 producer registers must not exceed MMA registers, "
+                f"got producer={registers.producer} mma={registers.mma}"
+            )
+    elif registers is not None:
+        errors.append(f"unexpected registers {registers!r} for arch {device_arch}")
+    return errors
+
+
+def validate_registers(
+    registers: Registers, inputs: FwdHeuristicInputs, q_stage: int
+) -> None:
+    """Check direction and budget of an explicit register split."""
+    errors = []
+    if _is_sm100_family(inputs.device_arch):
+        if not isinstance(registers, Sm100Registers):
+            errors.append(
+                f"SM{inputs.device_arch} requires Sm100Registers, got {registers!r}"
+            )
+        else:
+            errors.extend(_register_direction_errors(registers, inputs.device_arch))
+            budget = (
+                q_stage * registers.softmax
+                + registers.correction
+                + (3 - q_stage) * registers.other
+            )
+            if budget > SM100_MAX_REGISTER_BUDGET:
+                errors.append(
+                    "SM100 register budget exceeded: "
+                    f"q_stage*softmax + correction + (3-q_stage)*other = {budget} "
+                    f"> {SM100_MAX_REGISTER_BUDGET}"
+                )
+    elif inputs.device_arch == 90:
+        if not isinstance(registers, Sm90Registers):
+            errors.append(f"SM90 requires Sm90Registers, got {registers!r}")
+        else:
+            errors.extend(_register_direction_errors(registers, inputs.device_arch))
+            num_wg = 2
+            budget = num_wg * registers.mma + registers.producer
+            if budget > SM90_MAX_REGISTER_BUDGET:
+                errors.append(
+                    "SM90 register budget exceeded: "
+                    f"num_wg*mma + producer = {budget} > {SM90_MAX_REGISTER_BUDGET}"
+                )
+    elif registers is not None:
+        errors.append(
+            f"arch {inputs.device_arch} does not split registers by role, "
+            f"got {registers!r}"
+        )
+    if errors:
+        raise ValueError("; ".join(errors))
+
+
+# ---------------------------------------------------------------------------
+# SplitKV heuristic.
+# ---------------------------------------------------------------------------
+
+
+def num_splits_heuristic(
+    inputs: FwdHeuristicInputs, num_sm: int, max_splits: int = 128
+) -> int:
+    """Pick an exact SplitKV count from host-visible metadata only."""
+    if inputs.max_seqlen_k <= 0:
+        return 1
+    tile_n = 128
+    num_n_blocks = ceil_div(inputs.max_seqlen_k, tile_n)
+    num_m_blocks = ceil_div(inputs.total_q, 128)
+    if num_m_blocks <= 0:
+        return 1
+    # Only split when there is not enough parallelism to fill the device.
+    if num_m_blocks * num_n_blocks >= num_sm:
+        return 1
+    splits = min(max_splits, ceil_div(num_sm, num_m_blocks))
+    # Each split must own at least 2 KV blocks to amortize the reduction.
+    splits = min(splits, max(1, num_n_blocks // 2))
+    return max(1, splits)
+
+
+# ---------------------------------------------------------------------------
+# Selection.
+# ---------------------------------------------------------------------------
+
+
+@lru_cache(maxsize=None)
+def _select_fwd_config(inputs: FwdHeuristicInputs) -> FwdConfig:
+    device_arch = inputs.device_arch
+
+    use_2cta = is_2cta_eligible(inputs)
+    if inputs.requested_use_clc_scheduler and can_use_clc(inputs):
+        use_clc = True
+    else:
+        use_clc = False
+    is_static_persistent = can_use_static_persistent(inputs) and not use_clc
+    use_tma_o = can_use_tma_o(inputs)
+    use_s_ping_pong = can_use_s_ping_pong(inputs)
+
+    if device_arch in (100, 103, 110):
+        tuning = _sm100_tuning(inputs)
+        if tuning is None:
+            q_stage = 2
+        else:
+            q_stage = tuning[0]
+        if use_2cta and q_stage > 1:
+            q_stage -= 1
+        tile_m = 256 if use_2cta else 128
+        tile_n = inputs.head_dim_v if use_2cta else 128
+        mma_pv_is_rs = inputs.head_dim_v == 64
+        intra_wg_overlap = not use_2cta
+    elif device_arch == 90:
+        q_stage = 2
+        tile_m = 128
+        tile_n = 128
+        mma_pv_is_rs = False
+        intra_wg_overlap = True
+    else:
+        q_stage = 2
+        tile_m = 128
+        tile_n = 64 if inputs.head_dim_v <= 64 else 128
+        mma_pv_is_rs = False
+        intra_wg_overlap = True
+
+    if inputs.requested_tile_m is not None:
+        tile_m = inputs.requested_tile_m
+    if inputs.requested_tile_n is not None:
+        tile_n = inputs.requested_tile_n
+    if inputs.requested_mma_pv_is_rs is not None:
+        mma_pv_is_rs = inputs.requested_mma_pv_is_rs
+    if inputs.requested_intra_wg_overlap is not None:
+        intra_wg_overlap = inputs.requested_intra_wg_overlap
+
+    if inputs.requested_num_splits is not None:
+        num_splits = inputs.requested_num_splits
+    elif inputs.seqlen_k_per_split is not None:
+        num_splits = num_splits_heuristic(inputs, inputs.num_sms)
+    else:
+        num_splits = 1
+
+    registers = select_registers(inputs, tile_m, tile_n, q_stage, use_2cta)
+
+    return FwdConfig(
+        device_capacity=device_arch,
+        tile_m=tile_m,
+        tile_n=tile_n,
+        mma_pv_is_rs=mma_pv_is_rs,
+        intra_wg_overlap=intra_wg_overlap,
+        q_stage=q_stage,
+        use_clc_scheduler=use_clc,
+        is_static_persistent=is_static_persistent,
+        use_tma_o=use_tma_o,
+        use_s_ping_pong=use_s_ping_pong,
+        num_splits=num_splits,
+        use_2cta_instrs=use_2cta,
+        registers=registers,
+    )
+
+
+def select_fwd_config(inputs: FwdHeuristicInputs) -> FwdConfig:
+    """Select the default forward config and validate it before returning."""
+    config = _select_fwd_config(inputs)
+    validate_fwd_config(config, inputs)
+    return config
+
+
+# ---------------------------------------------------------------------------
+# Validation.
+# ---------------------------------------------------------------------------
+
+
+def validate_fwd_config(config: FwdConfig, inputs: FwdHeuristicInputs) -> None:
+    """Reject unsupported explicit configs instead of normalizing them."""
+    errors = []
+    if config.device_capacity != inputs.device_arch:
+        errors.append(
+            f"config device_capacity {config.device_capacity} does not match "
+            f"inputs device_arch {inputs.device_arch}"
+        )
+    if config.tile_m <= 0 or config.tile_n <= 0:
+        errors.append(
+            f"tile_m and tile_n must be positive, got {config.tile_m}x{config.tile_n}"
+        )
+    if config.q_stage <= 0:
+        errors.append(f"q_stage must be positive, got {config.q_stage}")
+    if config.num_splits < 1:
+        errors.append(f"num_splits must be >= 1, got {config.num_splits}")
+    if config.use_2cta_instrs and not is_2cta_eligible(inputs):
+        errors.append(
+            "2CTA instructions are not supported for this input "
+            "(arch, head dim, packed Q tile count, or disable_2cta)"
+        )
+    if config.use_clc_scheduler and not can_use_clc(inputs):
+        errors.append(
+            "CLC scheduler is not supported (needs SM100+, caller scheduler "
+            "metadata, and no tile-count semaphore)"
+        )
+    if config.is_static_persistent and not can_use_static_persistent(inputs):
+        errors.append(
+            "Static persistent scheduling is not supported (needs SM100+ and a "
+            "static tile count)"
+        )
+    if config.is_static_persistent and config.use_clc_scheduler:
+        errors.append("Static persistent scheduling excludes the CLC scheduler")
+    if config.use_tma_o and not can_use_tma_o(inputs):
+        errors.append(
+            "TMA O is not supported for varlen Q, score/mask mods, or this arch"
+        )
+    if config.use_s_ping_pong and not can_use_s_ping_pong(inputs):
+        errors.append(
+            "S ping-pong is not supported for this arch, block sparsity, or KV extent"
+        )
+    errors.extend(_register_direction_errors(config.registers, inputs.device_arch))
+    if errors:
+        raise ValueError("; ".join(errors))
+    validate_registers(config.registers, inputs, config.q_stage)
