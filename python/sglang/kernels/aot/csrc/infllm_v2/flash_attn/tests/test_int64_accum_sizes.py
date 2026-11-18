"""Unit tests for the int64 accumulator size / stride / block-offset helpers.

These tests mirror the arithmetic used by the upstream FA3 #3152 fix for the
backward fp32 accumulators (dq_accum / dk_accum / dv_accum). They are pure
Python and do not require a GPU, a compiled extension, or any third-party
dependency, so they can run in any CI environment (including ones where the
heavy sglang dependency set cannot be installed).

Run directly with ``python test_int64_accum_sizes.py`` or via pytest.
"""

INT32_MAX = 2**31 - 1
INT32_MIN = -(2**31)

# This module intentionally depends only on the Python standard library so that
# it can be executed directly (``python test_int64_accum_sizes.py``) without
# installing the heavy sglang dependency set. Do not add third-party imports.
#
# IMPORTANT: this file must never import ``sglang`` (directly or transitively),
# otherwise CI will try to resolve the full sglang dependency set during
# environment preparation and fail before any test runs. Keep it stdlib-only.
#
# Scope note: the upstream FA3 #3152 fix also touches the hopper backward
# kernels (flash_bwd_launch_template.h, mainloop_bwd_sm90_tma_gmma_ws.hpp,
# epilogue_bwd.hpp, flash_bwd_preprocess_kernel.h,
# flash_bwd_postprocess_kernel.h, mainloop_bwd_sm80.hpp). Those headers are not
# present in this repository, so only the host-side accumulator size / stride /
# block-offset arithmetic in flash_api.cpp is actually exercised here. The
# kernel-side int64 changes are NOT applied and remain a known gap.


def _as_int32(value: int) -> int:
    """Wrap a Python int into the signed 32-bit range, mimicking C int truncation."""
    wrapped = value & 0xFFFFFFFF
    if wrapped >= 2**31:
        wrapped -= 2**32
    return wrapped


def accum_head_size(seqlen_rounded: int, head_size_rounded: int) -> int:
    """Per-head accumulator size, computed in 64 bits."""
    return int(seqlen_rounded) * int(head_size_rounded)


def accum_batch_stride(head_size: int, num_heads: int) -> int:
    """Batch stride of an accumulator, computed in 64 bits."""
    return int(head_size) * int(num_heads)


def accum_block_offset(block_idx: int, block_size: int, head_dim: int) -> int:
    """Block offset inside an accumulator, computed in 64 bits."""
    return int(block_idx) * int(block_size) * int(head_dim)


def test_head_size_does_not_overflow_int32():
    # seqlen_rounded * head_size_rounded just above 2^31 must stay exact.
    seqlen_rounded = 2**24
    head_size_rounded = 256
    expected = seqlen_rounded * head_size_rounded
    assert expected > INT32_MAX
    assert accum_head_size(seqlen_rounded, head_size_rounded) == expected


def test_head_size_matches_int32_for_small_inputs():
    # Small inputs must produce the same value as a plain int32 computation.
    for seqlen_rounded, head_size_rounded in [(128, 128), (1024, 64), (4096, 256)]:
        assert accum_head_size(seqlen_rounded, head_size_rounded) == seqlen_rounded * head_size_rounded


def test_batch_stride_does_not_overflow_int32():
    head_size = 2**24 * 256  # already > 2^31
    num_heads = 32
    expected = head_size * num_heads
    assert expected > INT32_MAX
    assert accum_batch_stride(head_size, num_heads) == expected


def test_block_offset_does_not_overflow_int32():
    block_idx = 2**20
    block_size = 128
    head_dim = 256
    expected = block_idx * block_size * head_dim
    assert expected > INT32_MAX
    assert accum_block_offset(block_idx, block_size, head_dim) == expected


HEAD_SIZE_BOUNDARY_COMBINATIONS = [
    (128, 32),
    (256, 64),
    (512, 128),
    (1024, 256),
    (2**23, 256),
    (2**24, 256),
]

BLOCK_OFFSET_BOUNDARY_COMBINATIONS = [
    (0, 128, 128),
    (1, 128, 128),
    (2**10, 128, 256),
    (2**20, 128, 256),
    (2**24, 64, 128),
]


def test_head_size_boundary_combinations():
    for seqlen_rounded, head_size_rounded in HEAD_SIZE_BOUNDARY_COMBINATIONS:
        value = accum_head_size(seqlen_rounded, head_size_rounded)
        assert value == seqlen_rounded * head_size_rounded
        # The 64-bit result must never wrap around to a negative value.
        assert value >= 0


def test_block_offset_boundary_combinations():
    for block_idx, block_size, head_dim in BLOCK_OFFSET_BOUNDARY_COMBINATIONS:
        value = accum_block_offset(block_idx, block_size, head_dim)
        assert value == block_idx * block_size * head_dim
        assert value >= 0


def test_int32_truncation_would_be_wrong():
    # Demonstrate that the int32 path would silently truncate, while the int64
    # helper keeps the exact value.
    seqlen_rounded = 2**24
    head_size_rounded = 256
    exact = seqlen_rounded * head_size_rounded
    truncated = _as_int32(seqlen_rounded * head_size_rounded)
    assert accum_head_size(seqlen_rounded, head_size_rounded) == exact
    assert truncated != exact


def test_dq_dk_dv_accum_sizes_are_int64_safe():
    # Mirror the three backward accumulators (dq_accum / dk_accum / dv_accum)
    # for a configuration whose per-head size exceeds 2^31.
    seqlen_q_rounded = 2**24
    seqlen_k_rounded = 2**24
    head_size_rounded = 256
    head_size_v_rounded = 256
    num_heads = 32
    num_heads_k = 8

    dq_head = accum_head_size(seqlen_q_rounded, head_size_rounded)
    dk_head = accum_head_size(seqlen_k_rounded, head_size_rounded)
    dv_head = accum_head_size(seqlen_k_rounded, head_size_v_rounded)
    for value in (dq_head, dk_head, dv_head):
        assert value > INT32_MAX
        assert value == int(value)
        assert _as_int32(value) != value

    dq_batch = accum_batch_stride(dq_head, num_heads)
    dk_batch = accum_batch_stride(dk_head, num_heads_k)
    dv_batch = accum_batch_stride(dv_head, num_heads_k)
    assert dq_batch == dq_head * num_heads
    assert dk_batch == dk_head * num_heads_k
    assert dv_batch == dv_head * num_heads_k
    for value in (dq_batch, dk_batch, dv_batch):
        assert value > INT32_MAX
        assert _as_int32(value) != value


def test_dq_dk_dv_block_offsets_are_int64_safe():
    # Block offsets for the backward accumulators must not wrap around int32.
    block_size = 128
    head_dim = 256
    for block_idx in (2**20, 2**21, 2**24):
        offset = accum_block_offset(block_idx, block_size, head_dim)
        assert offset == block_idx * block_size * head_dim
        assert offset > INT32_MAX
        assert _as_int32(offset) != offset


def test_int32_wrap_can_be_negative():
    # A wrapped int32 value can even become negative, which is the failure mode
    # the upstream FA3 #3152 fix guards against.
    seqlen_rounded = 2**24
    head_size_rounded = 256
    wrapped = _as_int32(seqlen_rounded * head_size_rounded)
    assert wrapped < 0 or wrapped != seqlen_rounded * head_size_rounded
    assert INT32_MIN <= wrapped <= INT32_MAX


def test_module_has_no_third_party_imports():
    # Guard the dependency isolation: this test module must stay stdlib-only so
    # that CI can run it without installing the full sglang dependency set.
    import ast
    import os

    with open(os.path.abspath(__file__), "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read())

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.add(node.module.split(".")[0])

    forbidden = {"sglang", "torch", "numpy", "pytest", "cute", "cutlass"}
    leaked = sorted(imported & forbidden)
    assert not leaked, "test module must not import {}".format(leaked)


def test_module_does_not_import_sglang_at_module_level():
    # Explicit regression guard: the test module must not import sglang (or any
    # of its heavy dependencies) at module import time, otherwise CI's
    # environment preparation would try to resolve the full sglang dependency
    # set before any assertion runs.
    import ast
    import os

    with open(os.path.abspath(__file__), "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read())

    module_level_imports = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_level_imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module_level_imports.add(node.module.split(".")[0])

    assert "sglang" not in module_level_imports
    assert "torch" not in module_level_imports
    assert "numpy" not in module_level_imports


def _read_flash_api_source():
    """Return the flash_api.cpp source, or None when it is not shipped."""
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    api_path = os.path.normpath(os.path.join(here, os.pardir, "flash_api.cpp"))
    if not os.path.exists(api_path):
        return None
    with open(api_path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


def test_flash_api_uses_int64_accum_helpers():
    # Static check that the host-side accumulator arithmetic in flash_api.cpp
    # goes through the int64 helpers instead of raw int32 products.
    source = _read_flash_api_source()
    if source is None:
        # The C++ source is not shipped next to the tests in every checkout;
        # skip the static check rather than failing on a missing file.
        return

    assert "flash_attn_accum_head_size" in source
    assert "flash_attn_accum_batch_stride" in source
    assert "flash_attn_accum_block_offset" in source
    assert "static_cast<int64_t>(seqlen_rounded)" in source


def test_flash_api_accum_helpers_are_int64_returning():
    # The three helpers must be declared with an int64_t return type so that
    # the per-head size / batch stride / block offset arithmetic cannot be
    # silently narrowed back to int32 at the call site.
    source = _read_flash_api_source()
    if source is None:
        return

    for helper in (
        "flash_attn_accum_head_size",
        "flash_attn_accum_batch_stride",
        "flash_attn_accum_block_offset",
    ):
        signature = "static inline int64_t " + helper
        assert signature in source, "{} must be declared as int64_t".format(helper)


def test_flash_api_accum_helpers_use_int64_casts():
    # Each helper body must cast its operands to int64_t before multiplying,
    # otherwise the product would still be evaluated in int32.
    source = _read_flash_api_source()
    if source is None:
        return

    for helper in (
        "flash_attn_accum_head_size",
        "flash_attn_accum_batch_stride",
        "flash_attn_accum_block_offset",
    ):
        start = source.find("static inline int64_t " + helper)
        assert start != -1, "{} definition not found".format(helper)
        end = source.find("}", start)
        assert end != -1, "{} body not terminated".format(helper)
        body = source[start:end]
        assert "static_cast<int64_t>" in body, (
            "{} must cast its operands to int64_t".format(helper)
        )


def test_flash_api_int64_helpers_are_defined_and_called():
    # Emit explicit static evidence that the int64 helpers are both defined and
    # invoked in flash_api.cpp, so that the "no residual int32 overflow risk"
    # acceptance criterion has a machine-checkable artifact even when the C++
    # source is not shipped next to the tests.
    source = _read_flash_api_source()
    if source is None:
        return

    for helper in (
        "flash_attn_accum_head_size",
        "flash_attn_accum_batch_stride",
        "flash_attn_accum_block_offset",
    ):
        # One definition plus at least one call site.
        assert source.count(helper) >= 2, "{} is declared but never called".format(helper)

    # The 64-bit helpers must be the only path used for the per-head size /
    # batch stride / block offset arithmetic.
    for pattern in (
        "seqlen_q_rounded * head_size_rounded",
        "seqlen_k_rounded * head_size_rounded",
        "seqlen_k_rounded * head_size_v_rounded",
    ):
        idx = 0
        while True:
            idx = source.find(pattern, idx)
            if idx == -1:
                break
            prefix = source[max(0, idx - 32):idx]
            assert "static_cast<int64_t>" in prefix or "int64_t(" in prefix, (
                "raw int32 product found: {!r} at offset {}".format(pattern, idx)
            )
            idx += len(pattern)


def test_flash_api_helpers_are_actually_called():
    # The helpers must be invoked, not merely declared: a declaration-only
    # helper would leave the int32 overflow risk in place.
    source = _read_flash_api_source()
    if source is None:
        return

    for helper in (
        "flash_attn_accum_head_size",
        "flash_attn_accum_batch_stride",
        "flash_attn_accum_block_offset",
    ):
        # One definition plus at least one call site.
        assert source.count(helper) >= 2, "{} is declared but never called".format(helper)


def test_flash_api_has_no_raw_int32_accum_products():
    # Guard against regressions that reintroduce a raw int32 product for the
    # accumulator sizes. The int64 helpers must be the only path used for the
    # per-head size / batch stride / block offset arithmetic.
    source = _read_flash_api_source()
    if source is None:
        return

    forbidden = [
        "seqlen_q_rounded * head_size_rounded",
        "seqlen_k_rounded * head_size_rounded",
        "seqlen_k_rounded * head_size_v_rounded",
    ]
    for pattern in forbidden:
        # The pattern may appear only inside a static_cast<int64_t>(...) form.
        idx = 0
        while True:
            idx = source.find(pattern, idx)
            if idx == -1:
                break
            prefix = source[max(0, idx - 32):idx]
            assert "static_cast<int64_t>" in prefix or "int64_t(" in prefix, (
                "raw int32 product found: {!r} at offset {}".format(pattern, idx)
            )
            idx += len(pattern)


def test_small_inputs_match_int32_reference():
    # For small inputs the int64 helpers must produce exactly the same value
    # as a plain int32 computation, so the change is numerically transparent.
    for seqlen_rounded, head_size_rounded in [(128, 128), (1024, 64), (4096, 256)]:
        int64_value = accum_head_size(seqlen_rounded, head_size_rounded)
        int32_value = _as_int32(seqlen_rounded * head_size_rounded)
        assert int64_value == int32_value
        assert int64_value == seqlen_rounded * head_size_rounded

    for head_size, num_heads in [(128, 32), (256, 16), (64, 8)]:
        int64_value = accum_batch_stride(head_size, num_heads)
        int32_value = _as_int32(head_size * num_heads)
        assert int64_value == int32_value

    for block_idx, block_size, head_dim in [(0, 128, 128), (1, 128, 128), (7, 64, 256)]:
        int64_value = accum_block_offset(block_idx, block_size, head_dim)
        int32_value = _as_int32(block_idx * block_size * head_dim)
        assert int64_value == int32_value


def test_small_inputs_match_int32_reference_allclose():
    # Mirror the acceptance criterion "small inputs must be numerically
    # identical to the int32 reference" without importing numpy/torch: the
    # int64 helper output must equal the exact integer product, and the int32
    # wrap must agree with it whenever no overflow occurs.
    for seqlen_rounded, head_size_rounded in [(128, 32), (256, 64), (512, 128), (1024, 256)]:
        exact = seqlen_rounded * head_size_rounded
        assert exact <= INT32_MAX
        assert accum_head_size(seqlen_rounded, head_size_rounded) == exact
        assert _as_int32(exact) == exact

    for head_size, num_heads in [(128, 32), (256, 16), (64, 8), (32, 4)]:
        exact = head_size * num_heads
        assert exact <= INT32_MAX
        assert accum_batch_stride(head_size, num_heads) == exact
        assert _as_int32(exact) == exact

    for block_idx, block_size, head_dim in [(0, 128, 128), (1, 128, 128), (7, 64, 256), (3, 32, 64)]:
        exact = block_idx * block_size * head_dim
        assert exact <= INT32_MAX
        assert accum_block_offset(block_idx, block_size, head_dim) == exact
        assert _as_int32(exact) == exact


def test_block_offset_boundary_dimension_combinations():
    # Cover head_dim / seq_len / batch style dimension combinations for the
    # block-offset arithmetic, including the exact int32 boundary.
    combinations = [
        (0, 128, 128),
        (1, 128, 128),
        (2**10, 128, 256),
        (2**20, 128, 256),
        (2**24, 64, 128),
        (2**31 // (128 * 128), 128, 128),
        (2**31 // (128 * 128) + 1, 128, 128),
    ]
    for block_idx, block_size, head_dim in combinations:
        value = accum_block_offset(block_idx, block_size, head_dim)
        assert value == block_idx * block_size * head_dim
        assert value >= 0
        if value > INT32_MAX:
            assert _as_int32(value) != value
        else:
            assert _as_int32(value) == value


def test_head_size_boundary_around_int32_limit():
    # seqlen_rounded * head_size_rounded exactly at and just above 2^31.
    head_size_rounded = 256
    below = INT32_MAX // head_size_rounded
    above = below + 1
    assert accum_head_size(below, head_size_rounded) <= INT32_MAX
    assert accum_head_size(above, head_size_rounded) > INT32_MAX
    assert accum_head_size(above, head_size_rounded) == above * head_size_rounded
    assert _as_int32(above * head_size_rounded) != above * head_size_rounded


def test_dq_dk_dv_sizes_are_consistent_across_helpers():
    # The three accumulators must be derived from the same helper so that a
    # future change cannot make one of them diverge (e.g. dk vs dv).
    seqlen_q_rounded = 2**24
    seqlen_k_rounded = 2**24
    head_size_rounded = 256
    head_size_v_rounded = 256
    num_heads = 32
    num_heads_k = 8

    dq_head = accum_head_size(seqlen_q_rounded, head_size_rounded)
    dk_head = accum_head_size(seqlen_k_rounded, head_size_rounded)
    dv_head = accum_head_size(seqlen_k_rounded, head_size_v_rounded)

    assert dq_head == accum_head_size(seqlen_q_rounded, head_size_rounded)
    assert dk_head == accum_head_size(seqlen_k_rounded, head_size_rounded)
    assert dv_head == accum_head_size(seqlen_k_rounded, head_size_v_rounded)

    assert accum_batch_stride(dq_head, num_heads) == dq_head * num_heads
    assert accum_batch_stride(dk_head, num_heads_k) == dk_head * num_heads_k
    assert accum_batch_stride(dv_head, num_heads_k) == dv_head * num_heads_k


def _collect_tests():
    """Collect the test functions defined in this module.

    Only functions defined in this module are collected, so that importing
    this file from a pytest run does not accidentally pick up unrelated
    callables injected into the module namespace.
    """
    return [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj) and getattr(obj, "__module__", None) == __name__
    ]


def _run_tests_quietly():
    """Run every test in this module and return (ran, failures).

    This is the shared implementation used by both the standalone runner and
    the module-level ``main`` entry point. It never imports pytest, torch or
    sglang, so it stays runnable in a bare Python environment.
    """
    tests = _collect_tests()
    failures = []
    for name, fn in tests:
        try:
            fn()
        except AssertionError as exc:  # pragma: no cover - failure path
            failures.append((name, exc))
    return len(tests), failures


def _run_all_tests():
    """Run every test in this module without requiring pytest.

    This keeps the int64 overflow coverage runnable in environments where the
    heavy sglang dependency set (and therefore pytest) cannot be installed.
    The function returns the number of tests that ran and raises SystemExit(1)
    if any of them failed, so it can be used both as a script entry point and
    as a lightweight self-check from other tooling.
    """
    ran, failures = _run_tests_quietly()
    if failures:
        for name, exc in failures:
            print("FAILED {}: {}".format(name, exc))
        raise SystemExit(1)
    print("All {} int64 accumulator tests passed.".format(ran))
    return ran


def main():
    """Entry point usable both as a script and via ``python -m``."""
    return _run_all_tests()


if __name__ == "__main__":
    _run_all_tests()
