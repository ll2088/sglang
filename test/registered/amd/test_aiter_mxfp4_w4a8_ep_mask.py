# CPU-only tests for the expert-parallel gate_valid / expert_map logic used by
# the AITER MXFP4 W4A8/W4A16 MoE backends. These mirror the helpers in
# aiter_mxfp4_w4a8_moe.py but do not import aiter/triton, so they run on any CI
# runner that has torch installed.
#
# Run with a plain `python -m pytest` (no .venv, no ROCm, no aiter).
import pytest
import torch


def build_ep_expert_map(num_experts, num_local_experts, ep_rank, device):
    expert_map = torch.full((num_experts + 1,), -1, dtype=torch.int32, device=device)
    start = ep_rank * num_local_experts
    expert_map[start : start + num_local_experts] = torch.arange(
        num_local_experts, dtype=torch.int32, device=device
    )
    return expert_map


def gate_valid_from_topk(expert_map, topk_ids):
    return (expert_map[topk_ids.long()] >= 0).to(torch.int32).contiguous()


def test_gate_valid_mask_matches_module_helper_when_available():
    """When the real module is importable, the CPU mirror must agree with it."""
    try:
        from sglang.srt.layers.moe.fused_moe_triton.aiter_mxfp4_w4a8_moe import (
            build_ep_expert_map as real_build_ep_expert_map,
            build_gate_valid_mask as real_build_gate_valid_mask,
        )
    except Exception:  # pragma: no cover - depends on the CI image
        pytest.skip("aiter W4A8 module is not importable on this runner")

    device = torch.device("cpu")
    expert_map = real_build_ep_expert_map(8, 2, 1, device)
    topk_ids = torch.tensor([[0, 2], [3, 5], [8, 1]], dtype=torch.int32)
    assert torch.equal(
        real_build_gate_valid_mask(expert_map, topk_ids),
        gate_valid_from_topk(expert_map, topk_ids),
    )


def test_gate_valid_mask_all_local_and_all_remote():
    expert_map = build_ep_expert_map(8, 2, 0, torch.device("cpu"))
    local_ids = torch.tensor([[0, 1]], dtype=torch.int32)
    remote_ids = torch.tensor([[2, 7]], dtype=torch.int32)
    local_valid = gate_valid_from_topk(expert_map, local_ids)
    remote_valid = gate_valid_from_topk(expert_map, remote_ids)
    assert torch.equal(local_valid, torch.ones_like(local_valid))
    assert torch.equal(remote_valid, torch.zeros_like(remote_valid))


def supports_parallel_config(use_all2all_kernels, enable_eplb, dp_size):
    return not use_all2all_kernels and not enable_eplb and dp_size <= 1


def check_parallel_config(use_all2all_kernels, enable_eplb, dp_size):
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
def test_parallel_config_capability(
    use_all2all_kernels, enable_eplb, dp_size, supported
):
    assert (
        supports_parallel_config(use_all2all_kernels, enable_eplb, dp_size)
        is supported
    )


@pytest.mark.parametrize(
    "use_all2all_kernels,enable_eplb,dp_size,match",
    [
        (True, False, 1, "all-to-all"),
        (False, True, 1, "EPLB"),
        (False, False, 2, "dp_size"),
    ],
    ids=["all2all", "eplb", "dp_size_gt_1"],
)
def test_check_parallel_config_raises(use_all2all_kernels, enable_eplb, dp_size, match):
    with pytest.raises(NotImplementedError, match=match):
        check_parallel_config(use_all2all_kernels, enable_eplb, dp_size)


def test_check_parallel_config_accepts_naive_ep():
    check_parallel_config(False, False, 1)


def test_expert_map_marks_non_local_experts():
    num_experts, num_local_experts = 8, 2
    expert_map = build_ep_expert_map(
        num_experts, num_local_experts, 1, torch.device("cpu")
    )
    assert expert_map.shape == (num_experts + 1,)
    assert expert_map.dtype == torch.int32
    assert expert_map[2].item() == 0
    assert expert_map[3].item() == 1
    assert expert_map[0].item() == -1
    assert expert_map[4].item() == -1
    assert expert_map[num_experts].item() == -1


def test_gate_valid_mask_skips_non_local_gates():
    expert_map = build_ep_expert_map(8, 2, 1, torch.device("cpu"))
    topk_ids = torch.tensor([[0, 2], [3, 5]], dtype=torch.int32)
    gate_valid = gate_valid_from_topk(expert_map, topk_ids)
    expected = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
    assert torch.equal(gate_valid, expected)


def test_gate_valid_mask_sentinel_slot_is_invalid():
    num_experts = 8
    expert_map = build_ep_expert_map(num_experts, 2, 1, torch.device("cpu"))
    topk_ids = torch.tensor([[num_experts]], dtype=torch.int32)
    assert gate_valid_from_topk(expert_map, topk_ids).item() == 0


@pytest.mark.parametrize("ep_size", [2, 4])
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("hash_routing", [False, True])
@pytest.mark.parametrize("num_experts", [8, 16])
@pytest.mark.parametrize("topk", [1, 2])
@pytest.mark.parametrize("num_tokens", [1, 6])
@pytest.mark.parametrize("hidden", [1, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", ["cpu"])
def test_ep_partial_outputs_sum_to_full_moe_output(
    ep_size, seed, hash_routing, num_experts, topk, num_tokens, hidden, dtype, device
):
    """Per-rank partial outputs masked by gate_valid must sum to the full MoE
    output; dropping gate_valid would double count or zero rows."""
    assert num_experts % ep_size == 0
    num_local_experts = num_experts // ep_size
    torch.manual_seed(seed)
    dev = torch.device(device)

    if hash_routing:
        # Hash routing: deterministic per-token expert ids, like tid2eid.
        base = torch.arange(num_tokens, dtype=torch.int64, device=dev).unsqueeze(1)
        offs = torch.arange(topk, dtype=torch.int64, device=dev).unsqueeze(0)
        topk_ids = ((base + offs) % num_experts).to(torch.int32)
    else:
        topk_ids = torch.randint(
            0, num_experts, (num_tokens, topk), dtype=torch.int32, device=dev
        )
    topk_weights = torch.rand(num_tokens, topk, dtype=dtype, device=dev)
    expert_out = torch.randn(num_experts, num_tokens, hidden, dtype=dtype, device=dev)

    full = torch.zeros(num_tokens, hidden, dtype=dtype, device=dev)
    for t in range(num_tokens):
        for k in range(topk):
            full[t] += topk_weights[t, k] * expert_out[topk_ids[t, k], t]

    partial_sum = torch.zeros(num_tokens, hidden, dtype=dtype, device=dev)
    for rank in range(ep_size):
        expert_map = build_ep_expert_map(num_experts, num_local_experts, rank, dev)
        gate_valid = gate_valid_from_topk(expert_map, topk_ids).to(dtype)
        partial = torch.zeros(num_tokens, hidden, dtype=dtype, device=dev)
        for t in range(num_tokens):
            for k in range(topk):
                partial[t] += (
                    gate_valid[t, k]
                    * topk_weights[t, k]
                    * expert_out[topk_ids[t, k], t]
                )
        partial_sum += partial

    assert torch.allclose(partial_sum, full, atol=1e-6)
