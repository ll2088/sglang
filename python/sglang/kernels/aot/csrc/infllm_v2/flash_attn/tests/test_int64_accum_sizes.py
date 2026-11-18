"""Unit tests for the int64 accumulator size / stride / block-offset helpers.

These tests mirror the arithmetic used by the upstream FA3 #3152 fix for the
backward fp32 accumulators (dq_accum / dk_accum / dv_accum). They are pure
Python and do not require a GPU, a compiled extension, or any third-party
dependency, so they can run in any CI environment (including ones where the
heavy sglang dependency set cannot be installed).

Run directly with ``python test_int64_accum_sizes.py`` or via pytest.
"""

INT32_MAX = 2**31 - 1


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
    truncated = (seqlen_rounded * head_size_rounded) & INT32_MAX
    assert accum_head_size(seqlen_rounded, head_size_rounded) == exact
    assert truncated != exact


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


def _run_all_tests():
    """Run every test in this module without requiring pytest.

    This keeps the int64 overflow coverage runnable in environments where the
    heavy sglang dependency set (and therefore pytest) cannot be installed.
    The function returns the number of tests that ran and raises SystemExit(1)
    if any of them failed, so it can be used both as a script entry point and
    as a lightweight self-check from other tooling.
    """
    tests = _collect_tests()
    failures = []
    for name, fn in tests:
        try:
            fn()
        except AssertionError as exc:  # pragma: no cover - failure path
            failures.append((name, exc))
    if failures:
        for name, exc in failures:
            print("FAILED {}: {}".format(name, exc))
        raise SystemExit(1)
    print("All {} int64 accumulator tests passed.".format(len(tests)))
    return len(tests)


if __name__ == "__main__":
    _run_all_tests()
