import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.test.ci.ci_register import register_cuda_ci

if torch.version.cuda is not None:
    from sglang.kernels.ops.speculative.sampling import (
        tree_speculative_sampling_target_only,
    )
else:
    from sgl_kernel import tree_speculative_sampling_target_only

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")

test_cases = [
    (
        1,
        1,
        [3, -1, -1, 4, 5, 18, 11, -1, -1, -1, 12, 18],
        [[0, 3, 4, 5], [6, 10, 11, -1]],
        [3, 2],
    ),
    (
        0,  # threshold_single
        0,  # threshold_acc
        [3, -1, -1, 4, 5, 18, 11, -1, -1, -1, 12, 18],
        [[0, 3, 4, 5], [6, 10, 11, -1]],
        [3, 2],
    ),
]


@pytest.mark.parametrize(
    "threshold_single, threshold_acc, expected_predicts, expected_accept_index, expected_accept_token_num",
    test_cases,
)
def test_tree_speculative_sampling_target_only(
    threshold_single,
    threshold_acc,
    expected_predicts,
    expected_accept_index,
    expected_accept_token_num,
):
    """
    Tests the tree_speculative_sampling_target_only function using Pytest parameterization.
    """
    device = "cuda"

    candidates = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [7, 8, 9, 10, 11, 12],
        ],
        dtype=torch.int64,
        device=device,
    )
    retrive_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10, 11],
        ],
        dtype=torch.int64,
        device=device,
    )
    retrive_next_token = torch.tensor(
        [
            [1, 2, -1, 4, 5, -1],
            [4, 2, 3, -1, 5, -1],
        ],
        dtype=torch.int64,
        device=device,
    )
    retrive_next_sibling = torch.tensor(
        [
            [-1, 3, -1, -1, -1, -1],
            [-1, -1, -1, -1, 1, -1],
        ],
        dtype=torch.int64,
        device=device,
    )

    target_logits = torch.full((2, 6, 20), 1, dtype=torch.float32, device=device)
    target_logits[0, 0, 3] = 10
    target_logits[0, 3, 4] = 10
    target_logits[0, 4, 5] = 10
    target_logits[1, 0, 11] = 10
    target_logits[1, 4, 12] = 10

    for i in range(target_logits.shape[0]):
        for j in range(target_logits.shape[1]):
            if torch.max(target_logits[i, j]) < 10:
                target_logits[i, j, 18] = 10

    temperatures = torch.tensor([0.01, 0.01], dtype=torch.float32, device=device)
    bs, num_draft_tokens = candidates.shape
    num_spec_step = len(expected_accept_index[0])
    predict_shape = (len(expected_predicts),)

    predicts = torch.full(predict_shape, -1, dtype=torch.int32, device=device)
    accept_index = torch.full((bs, num_spec_step), -1, dtype=torch.int32, device=device)
    accept_token_num = torch.full((bs,), 0, dtype=torch.int32, device=device)

    expanded_temperature = temperatures.unsqueeze(1).unsqueeze(1)
    target_probs = F.softmax(target_logits / expanded_temperature, dim=-1)
    draft_probs = torch.full_like(target_probs, 0, dtype=torch.float32, device=device)
    coins = torch.rand(bs, num_draft_tokens, device=device, dtype=torch.float32)
    coins_for_final_sampling = torch.rand(bs, device=device).to(torch.float32)

    tree_speculative_sampling_target_only(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        uniform_samples=coins,
        uniform_samples_for_final_sampling=coins_for_final_sampling,
        target_probs=target_probs,
        draft_probs=draft_probs,
        threshold_single=threshold_single,
        threshold_acc=threshold_acc,
        deterministic=True,
    )

    assert predicts.tolist() == expected_predicts, (
        f"Predicts mismatch for thresholds ({threshold_single}, {threshold_acc})"
    )
    assert accept_index.tolist() == expected_accept_index, (
        f"Accept index mismatch for thresholds ({threshold_single}, {threshold_acc})"
    )
    assert accept_token_num.tolist() == expected_accept_token_num, (
        f"Accept token num mismatch for thresholds ({threshold_single}, {threshold_acc})"
    )


@pytest.mark.parametrize(
    "candidate_tokens,target_probabilities,coin,threshold_single,expected_token,expected_accept_token_num",
    [
        ([2], [0.0, 1.0, 0.0], 0.0, 0.0, 1, 0),
        ([2], [0.0, 1.0, 0.0], 0.0, 1.0, 1, 0),
        ([1, 2, 3], [0.0, 0.25, 0.0, 0.75], 0.25, 1.0, 3, 1),
        ([1], [0.0, 0.25, 0.75], 0.0, 1.0, 1, 1),
        ([1], [0.0, 1.0], 1.0 - 2**-24, 1.0, 1, 1),
    ],
    ids=[
        "zero-mass-zero-threshold",
        "zero-mass-default-threshold",
        "interior-boundary",
        "lower-endpoint",
        "upper-endpoint",
    ],
)
def test_target_only_sampling_cdf_boundaries(
    candidate_tokens,
    target_probabilities,
    coin,
    threshold_single,
    expected_token,
    expected_accept_token_num,
):
    num_candidates = len(candidate_tokens)
    candidates = torch.tensor(
        [[0, *candidate_tokens]], dtype=torch.int64, device="cuda"
    )
    num_draft_tokens = candidates.shape[1]
    retrive_index = torch.arange(
        num_draft_tokens, dtype=torch.int64, device="cuda"
    ).unsqueeze(0)
    retrive_next_token = torch.full_like(retrive_index, -1)
    retrive_next_token[0, 0] = 1
    retrive_next_sibling = torch.full_like(retrive_index, -1)
    if num_candidates > 1:
        retrive_next_sibling[0, 1:num_candidates] = torch.arange(
            2, num_candidates + 1, dtype=torch.int64, device="cuda"
        )

    target_probs = torch.zeros(
        (1, num_draft_tokens, len(target_probabilities)),
        dtype=torch.float32,
        device="cuda",
    )
    target_probs[0, 0] = torch.tensor(
        target_probabilities, dtype=torch.float32, device="cuda"
    )
    target_probs[0, 1:, 0] = 1.0
    draft_probs = torch.zeros_like(target_probs)
    predicts = torch.full((num_draft_tokens,), -1, dtype=torch.int32, device="cuda")
    accept_index = torch.full((1, 2), -1, dtype=torch.int32, device="cuda")
    accept_token_num = torch.zeros((1,), dtype=torch.int32, device="cuda")
    coins = torch.zeros((1, num_draft_tokens), dtype=torch.float32, device="cuda")
    coins[0, 0] = coin

    tree_speculative_sampling_target_only(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        uniform_samples=coins,
        uniform_samples_for_final_sampling=torch.zeros(
            (1,), dtype=torch.float32, device="cuda"
        ),
        target_probs=target_probs,
        draft_probs=draft_probs,
        threshold_single=threshold_single,
        threshold_acc=1.0,
        deterministic=True,
    )

    assert predicts[0].item() == expected_token
    assert accept_token_num.item() == expected_accept_token_num


def test_chain_speculative_sampling_one_variant_for_all_strides():
    '''Serving rebuilds the candidate and probability views every step; strides must not add variants.'''
    import triton

    from sglang.kernels.ops.speculative.reject_sampling import (
        chain_speculative_sampling_triton,
        speculative_sampling_classic_kernel,
    )

    caches = getattr(speculative_sampling_classic_kernel, 'device_caches', None)
    if caches is None:
        pytest.skip('this Triton has no JITFunction.device_caches')

    device = 'cuda'
    num_slots, vocab = 5, 64

    def rows_view(x, pad, fill):
        backing = torch.full(
            (x.shape[0], num_slots + pad), fill, device=device, dtype=x.dtype
        )
        backing[:, :num_slots] = x
        return backing[:, :num_slots]

    def vocab_view(x, stride_v):
        backing = torch.zeros(
            (x.shape[0], num_slots, vocab * stride_v), device=device, dtype=x.dtype
        )
        backing[:, :, ::stride_v] = x
        return backing[:, :, ::stride_v]

    def run(bs, row_pad=0, stride_v=1):
        g = torch.Generator(device=device).manual_seed(1234)
        cand = torch.randint(
            0, vocab, (bs, num_slots), device=device, dtype=torch.int64, generator=g
        )
        idx = torch.arange(bs * num_slots, device=device, dtype=torch.int64).view(
            bs, num_slots
        )
        uni = torch.rand(
            (bs, num_slots), device=device, dtype=torch.float32, generator=g
        )
        uni_final = torch.rand((bs,), device=device, dtype=torch.float32, generator=g)
        target = torch.softmax(
            torch.randn((bs, num_slots, vocab), device=device, generator=g), dim=-1
        )
        draft = torch.softmax(
            torch.randn((bs, num_slots, vocab), device=device, generator=g), dim=-1
        )
        if row_pad:
            cand = rows_view(cand, row_pad, 0)
            idx = rows_view(idx, row_pad, 0)
            uni = rows_view(uni, row_pad, 0.0)
        if stride_v != 1:
            target = vocab_view(target, stride_v)
            draft = vocab_view(draft, stride_v)

        predicts = torch.full((bs * num_slots,), -1, dtype=torch.int32, device=device)
        accept_index = torch.full((bs, num_slots), -1, dtype=torch.int32, device=device)
        accept_num = torch.zeros((bs,), dtype=torch.int32, device=device)
        chain_speculative_sampling_triton(
            predicts,
            accept_index,
            accept_num,
            cand,
            idx,
            None,
            None,
            uni,
            uni_final,
            target,
            draft,
            1.0,
            1.0,
            True,
        )
        return predicts, accept_index, accept_num

    ref = run(4)
    triton_device = triton.runtime.driver.active.get_current_device()
    cache_entry = caches.get(triton_device)
    if cache_entry is None:
        pytest.skip('no JIT device cache for the active device')
    try:
        variants = len(cache_entry[0])
    except (IndexError, TypeError):
        pytest.skip('unsupported JITFunction.device_caches layout')

    for bs, row_pad, stride_v in [
        (4, 3, 1),
        (4, 0, 2),
        (1, 2, 1),
        (7, 5, 2),
        (4, 0, 1),
    ]:
        out = run(bs, row_pad, stride_v)
        if bs == 4:
            for got, expected in zip(out, ref):
                assert torch.equal(got, expected)
    assert len(cache_entry[0]) == variants


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
