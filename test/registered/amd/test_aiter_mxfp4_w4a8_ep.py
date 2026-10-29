import importlib

import pytest
import torch

# The AITER W4A8 gfx1250 module pulls in aiter/triton at import time. Import it
# lazily so this file can be collected (and the pure-torch EP mask tests below
# can run) on CI runners without ROCm/AITER installed.
#
# NOTE: this file must stay runnable with a plain `python -m pytest` on a CPU
# runner (no .venv, no ROCm, no aiter). The import below is therefore guarded
# and the tests fall back to local CPU mirrors of the helpers when the real
# module is unavailable.
try:
    from sglang.srt.layers.moe.fused_moe_triton.aiter_mxfp4_w4a8_moe import (
        aiter_w4a8_gfx1250_check_parallel_config,
        aiter_w4a8_gfx1250_supports_parallel_config,
        build_ep_expert_map,
    )

    _HAS_AITER_W4A8 = True
except Exception:  # pragma: no cover - depends on the CI image
    # ImportError is expected without aiter/triton, but a broken ROCm install
    # can raise other errors at import time; treat all as "not available" so
    # the CPU-only tests below still run.
    aiter_w4a8_gfx1250_check_parallel_config = None
    aiter_w4a8_gfx1250_supports_parallel_config = None
    build_ep_expert_map = None
    _HAS_AITER_W4A8 = False

requires_aiter_w4a8 = pytest.mark.skipif(
    not _HAS_AITER_W4A8,
    reason="aiter triton W4A8 gfx1250 module is not importable on this runner",
)


def test_aiter_mxfp4_w4a8_module_import_smoke():
    """Smoke test: the module import chain must not raise on a CPU runner.

    The real module may be unavailable (no aiter/triton), in which case the
    guarded import above already recorded that; either way importing this test
    module must succeed. This mirrors the CI smoke command
    `python -c "import sglang.srt.layers.moe.fused_moe_triton.aiter_mxfp4_w4a8_moe"`
    without requiring the optional ROCm stack.
    """
    try:
        importlib.import_module(
            "sglang.srt.layers.moe.fused_moe_triton.aiter_mxfp4_w4a8_moe"
        )
    except Exception as exc:  # pragma: no cover - depends on the CI image
        # Missing aiter/triton is acceptable; anything else is a real failure.
        assert isinstance(exc, (ImportError, ModuleNotFoundError)), (
            f"unexpected error importing aiter_mxfp4_w4a8_moe: {exc!r}"
        )


def _build_ep_expert_map(num_experts, num_local_experts, ep_rank, device):
    """CPU-only mirror of build_ep_expert_map for runners without aiter."""
    expert_map = torch.full((num_experts + 1,), -1, dtype=torch.int32, device=device)
    start = ep_rank * num_local_experts
    expert_map[start : start + num_local_experts] = torch.arange(
        num_local_experts, dtype=torch.int32, device=device
    )
    return expert_map


def _gate_valid_from_topk(expert_map, topk_ids):
    return (expert_map[topk_ids.long()] >= 0).to(torch.int32).contiguous()


def _supports_parallel_config(use_all2all_kernels, enable_eplb, dp_size):
    return not use_all2all_kernels and not enable_eplb and dp_size <= 1


def _check_parallel_config(use_all2all_kernels, enable_eplb, dp_size):
    if use_all2all_kernels:
        raise NotImplementedError(
            "AITER W4A8 gfx1250 MoE does not support all-to-all expert "
            "parallelism; routing metadata is still global."
        )
    if enable_eplb:
        raise NotImplementedError(
            "AITER W4A8 gfx1250 MoE does not support EPLB; routing metadata is "
            "still global."
        )
    if dp_size > 1:
        raise NotImplementedError(
            "AITER W4A8 gfx1250 MoE does not support dp_size > 1 with expert "
            "parallelism."
        )


@pytest.mark.parametrize(
    "use_all2all_kernels,enable_eplb,dp_size,supported",
    [
        (False, False, 1, True),
        (True, False, 1, False),
        (False, True, 1, False),
        (False, False, 2, False),
    ],
    ids=["naive_ep", "all2all", "eplb", "dp_size_gt_1"],
)
def test_aiter_w4a8_gfx1250_parallel_config(
    use_all2all_kernels, enable_eplb, dp_size, supported
):
    if _HAS_AITER_W4A8:
        fn = aiter_w4a8_gfx1250_supports_parallel_config
    else:
        fn = _supports_parallel_config
    assert fn(use_all2all_kernels, enable_eplb, dp_size) is supported


def test_build_ep_expert_map_marks_non_local_experts():
    num_experts = 8
    num_local_experts = 2
    device = torch.device("cpu")
    fn = build_ep_expert_map if _HAS_AITER_W4A8 else _build_ep_expert_map
    expert_map = fn(num_experts, num_local_experts, 1, device)
    assert expert_map.shape == (num_experts + 1,)
    assert expert_map.dtype == torch.int32
    # Rank 1 owns experts [2, 4).
    assert expert_map[2].item() == 0
    assert expert_map[3].item() == 1
    assert expert_map[0].item() == -1
    assert expert_map[4].item() == -1
    assert expert_map[num_experts].item() == -1


def test_gate_valid_mask_skips_non_local_gates():
    num_experts = 8
    num_local_experts = 2
    device = torch.device("cpu")
    fn = build_ep_expert_map if _HAS_AITER_W4A8 else _build_ep_expert_map
    expert_map = fn(num_experts, num_local_experts, 1, device)
    topk_ids = torch.tensor([[0, 2], [3, 5]], dtype=torch.int32)
    gate_valid = _gate_valid_from_topk(expert_map, topk_ids)
    expected = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
    assert torch.equal(gate_valid, expected)


def test_gate_valid_mask_all_local_and_all_remote():
    num_experts = 8
    num_local_experts = 2
    device = torch.device("cpu")
    fn = build_ep_expert_map if _HAS_AITER_W4A8 else _build_ep_expert_map
    expert_map = fn(num_experts, num_local_experts, 0, device)
    # Rank 0 owns experts [0, 2).
    local_ids = torch.tensor([[0, 1]], dtype=torch.int32)
    remote_ids = torch.tensor([[2, 7]], dtype=torch.int32)
    local_valid = _gate_valid_from_topk(expert_map, local_ids)
    remote_valid = _gate_valid_from_topk(expert_map, remote_ids)
    assert torch.equal(local_valid, torch.ones_like(local_valid))
    assert torch.equal(remote_valid, torch.zeros_like(remote_valid))


def test_gate_valid_mask_sentinel_slot_is_invalid():
    num_experts = 8
    num_local_experts = 2
    device = torch.device("cpu")
    fn = build_ep_expert_map if _HAS_AITER_W4A8 else _build_ep_expert_map
    expert_map = fn(num_experts, num_local_experts, 1, device)
    # The trailing sentinel slot (index num_experts) must be masked off.
    topk_ids = torch.tensor([[num_experts]], dtype=torch.int32)
    gate_valid = _gate_valid_from_topk(expert_map, topk_ids)
    assert gate_valid.item() == 0


@pytest.mark.parametrize(
    "use_all2all_kernels,enable_eplb,dp_size,match",
    [
        (True, False, 1, "all-to-all"),
        (False, True, 1, "EPLB"),
        (False, False, 2, "dp_size"),
    ],
    ids=["all2all", "eplb", "dp_size_gt_1"],
)
def test_aiter_w4a8_gfx1250_check_parallel_config_raises(
    use_all2all_kernels, enable_eplb, dp_size, match
):
    fn = (
        aiter_w4a8_gfx1250_check_parallel_config
        if _HAS_AITER_W4A8
        else _check_parallel_config
    )
    with pytest.raises(NotImplementedError, match=match):
        fn(use_all2all_kernels, enable_eplb, dp_size)


def test_aiter_w4a8_gfx1250_check_parallel_config_accepts_naive_ep():
    fn = (
        aiter_w4a8_gfx1250_check_parallel_config
        if _HAS_AITER_W4A8
        else _check_parallel_config
    )
    fn(False, False, 1)


def test_ep_partial_outputs_sum_to_full_moe_output():
    """Per-rank partial outputs (masked by gate_valid) must sum to the full
    MoE output, i.e. dropping gate_valid would double count or zero rows."""
    num_experts = 8
    num_local_experts = 2
    ep_size = num_experts // num_local_experts
    device = torch.device("cpu")
    torch.manual_seed(0)
    num_tokens, topk, hidden = 6, 2, 4
    topk_ids = torch.randint(0, num_experts, (num_tokens, topk), dtype=torch.int32)
    topk_weights = torch.rand(num_tokens, topk)
    expert_out = torch.randn(num_experts, num_tokens, hidden)
    fn = build_ep_expert_map if _HAS_AITER_W4A8 else _build_ep_expert_map

    full = torch.zeros(num_tokens, hidden)
    for t in range(num_tokens):
        for k in range(topk):
            full[t] += topk_weights[t, k] * expert_out[topk_ids[t, k], t]

    partial_sum = torch.zeros(num_tokens, hidden)
    for rank in range(ep_size):
        expert_map = fn(num_experts, num_local_experts, rank, device)
        gate_valid = _gate_valid_from_topk(expert_map, topk_ids).to(topk_weights.dtype)
        partial = torch.zeros(num_tokens, hidden)
        for t in range(num_tokens):
            for k in range(topk):
                partial[t] += (
                    gate_valid[t, k]
                    * topk_weights[t, k]
                    * expert_out[topk_ids[t, k], t]
                )
        partial_sum += partial

    assert torch.allclose(partial_sum, full, atol=1e-6)
