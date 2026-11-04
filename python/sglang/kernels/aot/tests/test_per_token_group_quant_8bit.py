import itertools
import sys

import pytest
import torch

# This module is collected in CPU-only CI environments as well. Probe the
# optional runtime dependencies at import time and skip the whole module when
# they are unavailable, instead of failing collection.
_HAS_CUDA = torch.cuda.is_available()

if _HAS_CUDA:
    try:
        from sgl_kernel.test_utils import (
            assert_all_close_or_tiny_diff,
            create_per_token_group_quant_test_data,
        )

        from sglang.srt.utils import is_hip
        from sglang.test.kernels.per_token_group_quant import (
            sglang_per_token_group_quant_8bit,
            triton_per_token_group_quant_8bit,
        )

        _HAS_KERNELS = True
    except ImportError:
        _HAS_KERNELS = False
else:
    _HAS_KERNELS = False

pytestmark = pytest.mark.skipif(
    not _HAS_KERNELS,
    reason="requires CUDA and sgl_kernel/sglang quant kernels",
)

_is_hip = is_hip() if _HAS_KERNELS else False
fp8_type_ = torch.float8_e4m3fnuz if _is_hip else torch.float8_e4m3fn

_ROW_MAJOR = dict(
    column_major_scales=False,
    scale_tma_aligned=False,
    scale_ue8m0=False,
    fuse_silu_and_mul=False,
    masked_layout_mode=None,
)
_COLUMN_MAJOR_FLAGS = [
    dict(
        column_major_scales=True,
        scale_tma_aligned=False,
        scale_ue8m0=False,
        fuse_silu_and_mul=False,
        masked_layout_mode=None,
    ),
    dict(
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=False,
        fuse_silu_and_mul=False,
        masked_layout_mode=None,
    ),
    dict(
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
        fuse_silu_and_mul=False,
        masked_layout_mode=None,
    ),
]


def _supported(group_size, dst_dtype, flags):
    # int8 has no column-major scale layout; UE8M0 scales are only built for group 128.
    if dst_dtype == torch.int8 and flags["column_major_scales"]:
        return False
    return not flags["scale_ue8m0"] or group_size == 128


configs = [
    config
    for config in itertools.product(
        [1, 4, 127, 4096],  # num_tokens
        [128, 384, 1664, 7168],  # hidden_dim
        [16, 32, 64, 128],  # group_size
        [None],  # num_ranks
        [fp8_type_, torch.int8],  # dtype
        [_ROW_MAJOR, *_COLUMN_MAJOR_FLAGS],
    )
    if _supported(config[2], config[4], config[5])
] + list(
    itertools.product(
        [1, 4 * 8, 768 * 8],
        # TODO support more
        [2048],
        [128],
        [8, 16, 32, 48],
        [fp8_type_],
        [
            dict(
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=True,
                fuse_silu_and_mul=True,
                masked_layout_mode=None,
            ),
            dict(
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=True,
                fuse_silu_and_mul=True,
                masked_layout_mode="balanced",
            ),
            dict(
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=True,
                fuse_silu_and_mul=True,
                masked_layout_mode="imbalanced",
            ),
            dict(
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=True,
                fuse_silu_and_mul=True,
                masked_layout_mode="extreme",
            ),
        ],
    )
)


@pytest.mark.parametrize(
    "num_tokens, hidden_dim, group_size, num_ranks, dst_dtype, flags", configs
)
def test_per_token_group_quant_with_column_major(
    num_tokens,
    hidden_dim,
    group_size,
    num_ranks,
    dst_dtype,
    flags,
):
    print(
        f"{num_tokens=} {hidden_dim=} {group_size=} {num_ranks=} {dst_dtype=} {flags=}"
    )

    arch_major, _ = torch.cuda.get_device_capability(torch.cuda.current_device())
    if flags["scale_ue8m0"] and (arch_major <= 9):
        pytest.skip("Only Blackwell need ue8m0 fusion")

    x, masked_m = create_per_token_group_quant_test_data(
        num_tokens=num_tokens, hidden_dim=hidden_dim, num_ranks=num_ranks, flags=flags
    )

    execute_kwargs = dict(
        x=x,
        masked_m=masked_m,
        group_size=group_size,
        eps=1e-10,
        dst_dtype=dst_dtype,
        **{k: v for k, v in flags.items() if k not in ["masked_layout_mode"]},
    )

    def _postprocess(x_q, x_s):
        if masked_m is not None:
            print(f"Mask tokens after {masked_m} to be zero")
            for i in range(len(masked_m)):
                x_q[i, masked_m[i] :, :] = 0
                x_s[i, masked_m[i] :, :] = 0
        return x_q, x_s

    x_q_triton, x_s_triton = _postprocess(
        *triton_per_token_group_quant_8bit(**execute_kwargs)
    )
    x_q_sglang, x_s_sglang = _postprocess(
        *sglang_per_token_group_quant_8bit(**execute_kwargs)
    )

    try:
        assert_all_close_or_tiny_diff(x_q_triton, x_q_sglang)
        torch.testing.assert_close(
            x_s_triton.contiguous(),
            x_s_sglang.contiguous(),
            rtol=1e-3,
            atol=1e-5,
            msg=lambda message: message + f" {x_s_triton=} {x_s_sglang=}",
        )
    except AssertionError:
        print(
            f"{x.shape=} {x_q_triton.shape=} {x_s_triton.shape=} {x_q_sglang.shape=} {x_s_sglang.shape=}"
        )
        print(f"{x=}")
        print(f"{masked_m=}")
        print(f"{x_q_triton=}")
        print(f"{x_s_triton=}")
        print(f"{x_q_sglang=}")
        print(f"{x_s_sglang=}")

        raise


@pytest.mark.parametrize("num_tokens, hidden_dim, group_size", [(1, 128, 128), (4, 384, 128), (127, 1664, 128)])
@pytest.mark.parametrize("dst_dtype", [fp8_type_, torch.int8])
def test_per_token_group_quant_pair_cvt_consistency(num_tokens, hidden_dim, group_size, dst_dtype):
    """Pair cvt (sm_100+) must match the element-wise fallback bit for bit."""
    x, _ = create_per_token_group_quant_test_data(
        num_tokens=num_tokens, hidden_dim=hidden_dim, num_ranks=None, flags=_ROW_MAJOR
    )
    x_q, x_s = sglang_per_token_group_quant_8bit(
        x=x, masked_m=None, group_size=group_size, eps=1e-10, dst_dtype=dst_dtype, **_ROW_MAJOR
    )
    x_q_ref, x_s_ref = triton_per_token_group_quant_8bit(
        x=x, masked_m=None, group_size=group_size, eps=1e-10, dst_dtype=dst_dtype, **_ROW_MAJOR
    )
    assert_all_close_or_tiny_diff(x_q_ref, x_q)
    torch.testing.assert_close(x_s_ref.contiguous(), x_s.contiguous(), rtol=1e-3, atol=1e-5)


@pytest.mark.parametrize("dst_dtype", [fp8_type_, torch.int8])
def test_per_token_group_quant_saturation_boundary(dst_dtype):
    """Out-of-range and signed-zero inputs must saturate without UB.

    NaN is intentionally excluded: the saturating cvt keeps NaN while the
    clamp-then-cast fallback maps it to the max value, so the two gated paths
    are only required to agree on non-NaN inputs.
    """
    x = torch.tensor(
        [1e30, -1e30, float("inf"), float("-inf"), 0.0, -0.0] * 32,
        dtype=torch.float32,
        device="cuda",
    ).reshape(1, -1)
    x_q, x_s = sglang_per_token_group_quant_8bit(
        x=x, masked_m=None, group_size=128, eps=1e-10, dst_dtype=dst_dtype, **_ROW_MAJOR
    )
    assert torch.isfinite(x_s).all()
    if dst_dtype == fp8_type_:
        assert torch.isfinite(x_q.float()).all()
        # Saturation must clamp to the representable FP8 E4M3 range.
        assert x_q.float().abs().max() <= 448.0


def test_per_token_group_quant_zero_scale_safe_reciprocal():
    """An all-zero group has a zero scale and must quantize to zero, not NaN."""
    x = torch.zeros(4, 256, dtype=torch.bfloat16, device="cuda")
    x_q, x_s = sglang_per_token_group_quant_8bit(
        x=x, masked_m=None, group_size=128, eps=1e-10, dst_dtype=fp8_type_, **_ROW_MAJOR
    )
    assert torch.count_nonzero(x_q.view(torch.uint8)) == 0
    assert torch.count_nonzero(x_q.float() * x_s.repeat_interleave(128, dim=-1)) == 0


def test_merge_attn_states_fp8_end_to_end():
    """merge_attn_states output must match the pre-change reference."""
    merge_attn_states = pytest.importorskip(
        "sglang.srt.layers.attention.merge_attn_states",
        reason="merge_attn_states python binding not available",
    )
    merge_state_v2 = merge_attn_states.merge_state_v2

    torch.manual_seed(0)
    num_tokens, num_heads, head_size = 8, 4, 128
    v_a = torch.randn(num_tokens, num_heads, head_size, dtype=torch.bfloat16, device="cuda")
    v_b = torch.randn(num_tokens, num_heads, head_size, dtype=torch.bfloat16, device="cuda")
    s_a = torch.randn(num_tokens, num_heads, dtype=torch.float32, device="cuda")
    s_b = torch.randn(num_tokens, num_heads, dtype=torch.float32, device="cuda")
    v_merged = torch.empty_like(v_a)
    s_merged = torch.empty_like(s_a)
    merge_state_v2(v_a, s_a, v_b, s_b, v_merged, s_merged)

    p = torch.softmax(s_a, dim=-1)
    q = torch.softmax(s_b, dim=-1)
    ref = (v_a.float() * p.unsqueeze(-1) + v_b.float() * q.unsqueeze(-1)) / (p + q).unsqueeze(-1)
    torch.testing.assert_close(v_merged.float(), ref, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
