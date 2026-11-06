import pytest

from sglang.srt.layers.attention.dsa import fused_mqa_topk_plan as plan_mod


@pytest.mark.parametrize("topk", plan_mod.SUPPORTED_TOPKS)
def test_max_shape_is_one_launch(topk):
    plan = plan_mod.plan_fused_mqa_topk_prefill(
        plan_mod.MAX_ROWS, plan_mod.MAX_KEYS, topk, available=True
    )
    assert plan is not None
    assert plan.rows == plan_mod.MAX_ROWS
    assert plan.keys == plan_mod.MAX_KEYS
    assert plan.workspace_bytes == plan_mod.WORKSPACE_BYTES
    assert plan.dense_chunks > 1
    assert plan.dense_logits_bytes == plan_mod.MAX_ROWS * plan_mod.MAX_KEYS * 4


@pytest.mark.parametrize(
    "rows, keys, topk",
    [
        (64, plan_mod.MAX_KEYS, 512),  # topk not specialised
        (plan_mod.MAX_ROWS + 1, plan_mod.MAX_KEYS, 2048),  # too many rows
        (1024, plan_mod.MIN_KEYS - 1, 2048),  # below the fusion threshold
        (1024, plan_mod.MAX_KEYS + 1, 2048),  # too many keys
        (0, plan_mod.MAX_KEYS, 2048),  # empty chunk
    ],
)
def test_ineligible_shapes_fall_back_to_dense(rows, keys, topk):
    assert plan_mod.plan_fused_mqa_topk_prefill(rows, keys, topk, available=True) is None
    ok, reason = plan_mod.fusion_eligible(rows, keys, topk)
    assert not ok and reason != "eligible"


def test_unavailable_kernel_keeps_dense_path(monkeypatch):
    assert plan_mod.fused_mqa_topk_available(lambda: False) is False
    monkeypatch.setenv("SGLANG_DSA_FUSED_PREFILL_TOPK", "1")
    assert plan_mod.fused_mqa_topk_available(lambda: False) is False
    assert plan_mod.fused_mqa_topk_available(lambda: True) is True
    assert plan_mod.plan_fused_mqa_topk_prefill(8192, 1 << 18, 2048) is not None


def test_dense_chunks_shrink_with_budget():
    assert plan_mod.dense_logits_chunks(0, 4096) == 0
    assert plan_mod.dense_logits_chunks(128, 1 << 20) == 1
    assert plan_mod.dense_logits_chunks(16384, 1 << 20) == 128
    assert plan_mod.dense_logits_chunks(16384, 1 << 20) > plan_mod.dense_logits_chunks(
        16384, 1 << 14
    )


def test_row_rescan_bounds():
    assert plan_mod.row_rescan_required(2047, 2048)
    assert not plan_mod.row_rescan_required(2048, 2048)
    assert not plan_mod.row_rescan_required(plan_mod.CANDIDATES_PER_ROW, 2048)
    assert plan_mod.row_rescan_required(plan_mod.CANDIDATES_PER_ROW + 1, 2048)
