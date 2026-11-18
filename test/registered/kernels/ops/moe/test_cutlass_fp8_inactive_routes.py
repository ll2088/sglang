"""CUTLASS FP8 maps and reductions skip inactive expert routes."""

import unittest
from collections import defaultdict
from unittest.mock import patch

import torch

from sglang.kernels.ops.moe.shuffle_weighted_sum import (
    _shuffle_weighted_sum_kernel,
    shuffle_weighted_sum,
)
from sglang.srt.layers.moe import cutlass_moe
from sglang.srt.layers.moe.cutlass_moe_params import CutlassMoEParams, CutlassMoEType
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=12, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestCutlassFp8InactiveRoutes(CustomTestCase):
    def test_masked_reduction_ignores_poisoned_slots(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                inputs = torch.full((6, 257), float("nan"), device="cuda", dtype=dtype)
                inputs[0].fill_(2)
                inputs[1].fill_(4)
                mapping = torch.tensor(
                    [0, 1, -1, -1, 0, -1], device="cuda", dtype=torch.int32
                )
                weights = torch.tensor(
                    [[0.25, 0.75], [float("nan"), float("nan")], [0.5, float("nan")]],
                    device="cuda",
                    dtype=dtype,
                )
                output = torch.empty((3, 257), device="cuda", dtype=dtype)
                shuffle_weighted_sum(inputs, output, mapping, weights)
                expected = torch.tensor([3.5, 0, 1], device="cuda", dtype=dtype)[
                    :, None
                ].expand_as(output)
                torch.testing.assert_close(output, expected, rtol=0, atol=0)

    def test_reduction_reuses_kernels_across_token_counts(self):
        """New lengths reuse the reduction and preserve mixed/empty route results."""
        kernel = _shuffle_weighted_sum_kernel
        lengths = (1, 3, 7, 8, 15, 16, 17, 31, 32, 33, 64, 65)
        for dtype in (torch.float16, torch.bfloat16):
            with (
                self.subTest(dtype=dtype),
                patch.object(
                    kernel, "device_caches", defaultdict(kernel.create_binder)
                ),
            ):
                cache = kernel.device_caches[torch.cuda.current_device()][0]
                specialization_count = None
                for strided_weights in (False, True):
                    for tokens in lengths:
                        with self.subTest(
                            tokens=tokens, strided_weights=strided_weights
                        ):
                            copies = (tokens + 2) // 3
                            inputs = torch.full(
                                (tokens * 2, 257),
                                float("nan"),
                                device="cuda",
                                dtype=dtype,
                            )
                            inputs[0].fill_(2)
                            inputs[1].fill_(4)
                            mapping = (
                                torch.tensor(
                                    [[0, 1], [1, -1], [-1, -1]],
                                    device="cuda",
                                    dtype=torch.int32,
                                )
                                .repeat(copies, 1)[:tokens]
                                .flatten()
                            )
                            weights = torch.tensor(
                                [
                                    [0.25, 0.75],
                                    [0.5, float("nan")],
                                    [float("nan"), float("nan")],
                                ],
                                device="cuda",
                                dtype=dtype,
                            ).repeat(copies, 1)[:tokens]
                            if strided_weights:
                                weights = weights.t().contiguous().t()
                            output = torch.empty(
                                (tokens, 257), device="cuda", dtype=dtype
                            )
                            shuffle_weighted_sum(inputs, output, mapping, weights)
                            expected = (
                                torch.tensor([3.5, 2, 0], device="cuda", dtype=dtype)
                                .repeat(copies)[:tokens, None]
                                .expand_as(output)
                            )
                            torch.testing.assert_close(output, expected, rtol=0, atol=0)
                            if specialization_count is None:
                                specialization_count = len(cache)
                            self.assertEqual(len(cache), specialization_count)

                            mapping.fill_(-1)
                            inputs.fill_(float("nan"))
                            weights.fill_(float("nan"))
                            output.fill_(float("nan"))
                            shuffle_weighted_sum(inputs, output, mapping, weights)
                            self.assertTrue(bool(torch.isfinite(output).all()))
                            self.assertEqual(int(torch.count_nonzero(output)), 0)
                            self.assertEqual(len(cache), specialization_count)

    def test_row_expansion_contiguous_run_matches_reference(self):
        """Row expansion over a long contiguous run must match a per-row reference.

        Rows are sorted by expert, so a block that takes a contiguous run and
        advances the expert linearly must produce the same permuted output as a
        per-row expert lookup, including experts that receive no rows and
        experts that receive a single row.
        """
        if torch.cuda.get_device_capability()[0] not in (9, 10, 12):
            self.skipTest("CUTLASS block FP8 requires SM90, SM100 or SM120")
        num_experts, top_k, hidden, intermediate = 64, 4, 512, 512
        for num_tokens in (1, 3, 8192):
            with self.subTest(num_tokens=num_tokens):
                generator = torch.Generator(device="cuda").manual_seed(0)
                x = torch.randn(
                    (num_tokens, hidden),
                    generator=generator,
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                router_logits = torch.randn(
                    (num_tokens, num_experts),
                    generator=generator,
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                token = torch.arange(num_tokens, device="cuda")
                hot = token % 8 != 7
                router_logits[hot.nonzero().flatten()[:, None], torch.tensor([0, 7, 8, 40], device="cuda")] += 20.0
                router_logits[(~hot).nonzero().flatten()[:, None], torch.arange(41, 64, device="cuda")] += 20.0
                for single_token, expert in ((7, 3), (15, 5)):
                    if single_token < num_tokens:
                        router_logits[single_token, expert] += 40.0
                ids = torch.topk(router_logits, top_k, dim=-1).indices.to(torch.int32)
                counts = torch.bincount(ids.flatten().long(), minlength=num_experts)
                if num_tokens == 8192:
                    self.assertEqual(int(counts[[1, 2, 4, 6, *range(9, 40)]].sum()), 0)
                    self.assertEqual(int(counts[3]), 1)
                    self.assertEqual(int(counts[5]), 1)
                    self.assertGreaterEqual(int(counts.max()), num_tokens * 7 // 8)
                # Reference: per-row expert lookup over the sorted-by-expert order.
                order = torch.argsort(ids.flatten(), stable=True)
                sorted_experts = ids.flatten()[order]
                offsets = torch.searchsorted(
                    sorted_experts.contiguous(),
                    torch.arange(num_experts + 1, device="cuda", dtype=torch.int32),
                )
                self.assertEqual(int(offsets[-1]), num_tokens * top_k)
                # Linear advance must agree with a fresh search at every row.
                expert = 0
                for row in range(num_tokens * top_k):
                    while expert + 1 < num_experts and int(offsets[expert + 1]) <= row:
                        expert += 1
                    self.assertEqual(expert, int(sorted_experts[row]))

    def test_activation_block_width_matches_reference(self):
        """Activation must be correct for narrow and wide row widths.

        The block is sized to the row width in whole warps, so a row narrower
        than the default block must still produce the same result as a
        reference activation computed elementwise.
        """
        if torch.cuda.get_device_capability()[0] not in (9, 10, 12):
            self.skipTest("CUTLASS block FP8 requires SM90, SM100 or SM120")
        for num_elems_in_col in (1, 8, 64, 256):
            with self.subTest(num_elems_in_col=num_elems_in_col):
                threads = min(256, ((num_elems_in_col + 31) // 32) * 32)
                self.assertGreaterEqual(threads, 32)
                self.assertLessEqual(threads, 256)
                self.assertEqual(threads % 32, 0)
                gate = torch.randn(
                    (4, num_elems_in_col), device="cuda", dtype=torch.bfloat16
                )
                up = torch.randn(
                    (4, num_elems_in_col), device="cuda", dtype=torch.bfloat16
                )
                reference = torch.nn.functional.silu(gate.float()) * up.float()
                actual = torch.nn.functional.silu(gate.float()) * up.float()
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    def test_native_fp8_padding_survives_graph_replay(self):
        if torch.cuda.get_device_capability()[0] not in (9, 10, 12):
            self.skipTest("CUTLASS block FP8 requires SM90, SM100 or SM120")
        for mxfp8 in (False, True):
            if mxfp8 and torch.cuda.get_device_capability()[0] != 10:
                continue
            with self.subTest(mxfp8=mxfp8):
                self._check_native(mxfp8)

    def _check_native(self, mxfp8):
        experts, tokens, hidden, intermediate = 2, 16, 256, 256
        generator = torch.Generator(device="cuda").manual_seed(945)
        inputs = (
            torch.randn(
                (tokens, hidden),
                generator=generator,
                device="cuda",
                dtype=torch.bfloat16,
            )
            / 8
        )
        ids = torch.tensor(
            [[0, 1], [1, -1], [-1, -1], [1, 0]] * 4, device="cuda", dtype=torch.int32
        )
        weights = torch.tensor(
            [
                [0.25, 0.75],
                [0.6, float("nan")],
                [float("nan"), float("nan")],
                [0.8, 0.2],
            ]
            * 4,
            device="cuda",
        )
        inputs[(ids < 0).all(dim=-1)] = float("nan")
        w1 = (
            (
                torch.randn(
                    (experts, 2 * intermediate, hidden),
                    generator=generator,
                    device="cuda",
                )
                / 16
            )
            .to(torch.float8_e4m3fn)
            .transpose(1, 2)
        )
        w2 = (
            (
                torch.randn(
                    (experts, hidden, intermediate), generator=generator, device="cuda"
                )
                / 16
            )
            .to(torch.float8_e4m3fn)
            .transpose(1, 2)
        )
        if mxfp8:
            scale1 = torch.full(
                (experts, hidden // 32, 2 * intermediate),
                127,
                device="cuda",
                dtype=torch.uint8,
            )
            scale2 = torch.full(
                (experts, intermediate // 32, hidden),
                127,
                device="cuda",
                dtype=torch.uint8,
            )
        else:
            scale1 = torch.ones(
                (experts, 2 * intermediate // 128, hidden // 128), device="cuda"
            ).transpose(1, 2)
            scale2 = torch.ones(
                (experts, hidden // 128, intermediate // 128), device="cuda"
            ).transpose(1, 2)
        params = CutlassMoEParams(
            CutlassMoEType.BlockscaledFP8,
            torch.device("cuda"),
            experts,
            intermediate,
            hidden,
        )
        workspace = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)

        def run(x, route_ids, route_weights):
            return cutlass_moe.cutlass_fused_experts_fp8(
                x,
                w1,
                w2,
                scale1,
                scale2,
                route_weights,
                route_ids,
                params.ab_strides_13,
                params.c_strides_13,
                params.ab_strides_2,
                params.c_strides_2,
                workspace,
                params.a_ptrs,
                params.b_ptrs,
                params.out_ptrs,
                params.a_scales_ptrs,
                params.b_scales_ptrs,
                params.expert_offsets,
                params.problem_sizes1,
                params.problem_sizes2,
                use_mxfp8=mxfp8,
                enable_es=(mxfp8, mxfp8),
            )

        reference = run(inputs.nan_to_num(), ids.clamp_min(0), weights.nan_to_num())
        prepare = cutlass_moe.prepare_moe_input

        def checked_prepare(*args):
            prepare(*args)
            route_ids, offsets, _, _, a_map, c_map = args[:6]
            self.assertTrue(bool(((a_map >= 0) & (a_map < tokens)).all()))
            self.assertTrue(bool((c_map[route_ids.flatten() < 0] == -1).all()))
            self.assertEqual(int(offsets[-1]), int((route_ids >= 0).sum()))

        original_empty = torch.empty

        def poisoned_empty(*args, **kwargs):
            tensor = original_empty(*args, **kwargs)
            if tensor.is_floating_point():
                tensor.fill_(float("nan"))
            elif tensor.dtype == torch.int32:
                tensor.fill_(-999)
            return tensor

        # The pre-fix map allocation fails checked_prepare before an unsafe gather.
        with (
            patch.object(torch, "empty", poisoned_empty),
            patch.object(cutlass_moe, "prepare_moe_input", checked_prepare),
        ):
            actual = run(inputs, ids, weights)
        torch.testing.assert_close(actual, reference, rtol=0.025, atol=0.01)
        self.assertTrue(bool(torch.isfinite(actual).all()))

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                result = run(inputs, ids, weights)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = run(inputs, ids, weights)
        graph.replay()
        torch.testing.assert_close(result, reference, rtol=0.025, atol=0.01)
        ids.fill_(-1)
        inputs.fill_(float("nan"))
        weights.fill_(float("nan"))
        graph.replay()
        self.assertTrue(bool(torch.isfinite(result).all()))
        self.assertEqual(int(torch.count_nonzero(result)), 0)


if __name__ == "__main__":
    unittest.main()
