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


@pytest.mark.parametrize("topk", plan_mod.SUPPORTED_TOPKS)
@pytest.mark.parametrize(
    "num_candidates",
    [
        0,
        1,
        plan_mod.CANDIDATES_PER_ROW - 1,
        plan_mod.CANDIDATES_PER_ROW,
        plan_mod.CANDIDATES_PER_ROW + 1,
    ],
)
def test_rescan_window_matches_dense_contract(topk, num_candidates):
    # Too few candidates (the gate missed the boundary) or more than the
    # selection scratch holds both force the exact fp32 rescan.
    expected = not (topk <= num_candidates <= plan_mod.CANDIDATES_PER_ROW)
    assert plan_mod.row_rescan_required(num_candidates, topk) is expected


@pytest.mark.parametrize("topk", plan_mod.SUPPORTED_TOPKS)
def test_boundary_shapes(topk):
    assert plan_mod.fusion_eligible(plan_mod.MAX_ROWS, plan_mod.MAX_KEYS, topk) == (
        True,
        "eligible",
    )
    assert not plan_mod.fusion_eligible(
        plan_mod.MAX_ROWS + 1, plan_mod.MAX_KEYS, topk
    )[0]
    assert not plan_mod.fusion_eligible(
        plan_mod.MAX_ROWS, plan_mod.MAX_KEYS + 1, topk
    )[0]
    assert not plan_mod.fusion_eligible(
        plan_mod.MAX_ROWS, plan_mod.MIN_KEYS - 1, topk
    )[0]


def test_workspace_is_fixed_and_smaller_than_dense_logits():
    # Scratch is sized once for the maximum shape and reused across layers, so
    # it is shape independent and stays CUDA-graph capture safe.
    small = plan_mod.plan_fused_mqa_topk_prefill(
        4096, plan_mod.MIN_KEYS, 2048, available=True
    )
    large = plan_mod.plan_fused_mqa_topk_prefill(
        plan_mod.MAX_ROWS, plan_mod.MAX_KEYS, 2048, available=True
    )
    assert small.workspace_bytes == large.workspace_bytes == plan_mod.WORKSPACE_BYTES
    assert large.dense_logits_bytes == plan_mod.MAX_ROWS * plan_mod.MAX_KEYS * 4
    assert large.dense_logits_bytes > plan_mod.WORKSPACE_BYTES
    assert large.dense_chunks == plan_mod.dense_logits_chunks(
        plan_mod.MAX_ROWS, plan_mod.MAX_KEYS
    )


def test_resolve_prefill_plan_reports_reason():
    plan, reason = plan_mod.resolve_prefill_plan(
        plan_mod.MAX_ROWS, plan_mod.MAX_KEYS, 2048, available=True
    )
    assert plan is not None and reason == "eligible"

    plan, reason = plan_mod.resolve_prefill_plan(
        plan_mod.MAX_ROWS + 1, plan_mod.MAX_KEYS, 2048, available=True
    )
    assert plan is None and "exceeds the single-launch limit" in reason

    plan, reason = plan_mod.resolve_prefill_plan(
        plan_mod.MAX_ROWS, plan_mod.MAX_KEYS, 2048, available=False
    )
    assert plan is None and reason == "fused kernel unavailable or disabled"


def test_prefill_plan_hook_degrades_to_dense_on_failure(monkeypatch):
    """A failing probe/plan must return None, never raise into the prefill."""
    from sglang.srt.layers.attention.dsa.dsa_topk_backend import DSATopKBackend

    def _boom(*args, **kwargs):
        raise RuntimeError("kernel probe failed")

    monkeypatch.setattr(plan_mod, "plan_fused_mqa_topk_prefill", _boom)
    plan = DSATopKBackend.SGL_KERNEL.fused_prefill_plan(
        plan_mod.MAX_ROWS, plan_mod.MAX_KEYS, 2048
    )
    assert plan is None


def test_prefill_plan_hook_matches_plan_module(monkeypatch):
    """The hook is a thin pass-through: same shape in, same plan out."""
    from sglang.srt.layers.attention.dsa.dsa_topk_backend import DSATopKBackend

    expected = plan_mod.plan_fused_mqa_topk_prefill(
        plan_mod.MAX_ROWS, plan_mod.MAX_KEYS, 2048, available=True
    )
    got = DSATopKBackend.SGL_KERNEL.fused_prefill_plan(
        plan_mod.MAX_ROWS, plan_mod.MAX_KEYS, 2048
    )
    assert got == expected


if __name__ == "__main__":
    # Run with the interpreter that is actually on PATH (`python <file>`)
    # instead of relying on a hard-coded interpreter name.
    import sys

    raise SystemExit(pytest.main([__file__, "-q"]))
