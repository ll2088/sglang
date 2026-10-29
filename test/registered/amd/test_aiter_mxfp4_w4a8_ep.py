import pytest
import torch

from sglang.srt.layers.moe.fused_moe_triton.aiter_mxfp4_w4a8_moe import (
    aiter_w4a8_gfx1250_supports_parallel_config,
    build_ep_expert_map,
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
    assert (
        aiter_w4a8_gfx1250_supports_parallel_config(
            use_all2all_kernels, enable_eplb, dp_size
        )
        is supported
    )


def test_build_ep_expert_map_marks_non_local_experts():
    num_experts = 8
    num_local_experts = 2
    device = torch.device("cpu")
    expert_map = build_ep_expert_map(num_experts, num_local_experts, 1, device)
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
    expert_map = build_ep_expert_map(num_experts, num_local_experts, 1, device)
    topk_ids = torch.tensor([[0, 2], [3, 5]], dtype=torch.int32)
    gate_valid = (expert_map[topk_ids.long()] >= 0).to(torch.int32).contiguous()
    expected = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
    assert torch.equal(gate_valid, expected)
