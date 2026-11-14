"""Dependency-free self-check for the MXFP4 Engram layout module.

This script exercises the same behaviours as ``test_engram_mxfp4.py`` but
uses only the standard library, so it can run even when the full test
dependency set (pytest, torch, ...) is unavailable during environment
preparation. It is intentionally importable as a pytest module too: the
``test_*`` functions below are plain assertions and pytest will collect them
when the environment is healthy.

Run directly with::

    python tests/layers/test_engram_mxfp4_selfcheck.py
"""

from __future__ import annotations

import importlib.util
import os
import sys

# Make ``sglang`` importable when running this file directly from a checkout.
# This module is stdlib-only on purpose: it must be collectable and runnable
# even when the full sglang dependency chain (torch, triton, ...) is not
# installed, so CI environment preparation failures cannot block it.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PYTHON_DIR = os.path.join(_REPO_ROOT, "python")
if os.path.isdir(_PYTHON_DIR) and _PYTHON_DIR not in sys.path:
    sys.path.insert(0, _PYTHON_DIR)


def _load_engram_mxfp4():
    """Import the layout module without requiring the full sglang package.

    ``engram_mxfp4`` is stdlib-only, so it is always loaded directly from its
    file path. This never triggers the heavy sglang dependency chain (torch,
    triton, ...), which keeps the self-check runnable during a minimal CI
    environment preparation where ``pip install -e python`` may fail. There is
    deliberately no ``from sglang...`` fallback: importing the package would
    pull in the very dependencies this entry point exists to avoid.
    """
    module_path = os.path.join(
        _PYTHON_DIR, "sglang", "srt", "layers", "engram_mxfp4.py"
    )
    if not os.path.isfile(module_path):
        raise ImportError(f"engram_mxfp4 module not found at {module_path!r}")
    spec = importlib.util.spec_from_file_location("engram_mxfp4", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load engram_mxfp4 from {module_path!r}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_engram_mxfp4 = _load_engram_mxfp4()

E2M1_VALUES = _engram_mxfp4.E2M1_VALUES
EngramLayout = _engram_mxfp4.EngramLayout
EngramLayoutError = _engram_mxfp4.EngramLayoutError
allocate_row_buffer = _engram_mxfp4.allocate_row_buffer
decode_e2m1_nibble = _engram_mxfp4.decode_e2m1_nibble
decode_e2m1_packed = _engram_mxfp4.decode_e2m1_packed
dequantize_row = _engram_mxfp4.dequantize_row
lookup_rows = _engram_mxfp4.lookup_rows
pack_e2m1 = _engram_mxfp4.pack_e2m1
row_bytes_for = _engram_mxfp4.row_bytes_for

LAYER_IDS = (17, 19)


def _mxfp4_layout(head_dim: int = 256) -> EngramLayout:
    return EngramLayout.from_metadata(
        head_dim, LAYER_IDS, {"engram_embed_quant": "mxfp4"}
    )


def _expect_error(fn, *args, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except EngramLayoutError:
        return
    raise AssertionError(f"expected EngramLayoutError from {fn!r}")


# --- layout parsing -------------------------------------------------------


def test_default_layout_is_fp8() -> None:
    layout = EngramLayout.from_metadata(256, LAYER_IDS, None)
    assert layout.quant == "fp8"
    assert layout.packed_fp4 is False
    assert layout.row_bytes == 256
    assert layout.num_blocks == 8
    assert layout.total_row_bytes() == 264


def test_mxfp4_layout_halves_row_width() -> None:
    layout = _mxfp4_layout()
    assert layout.quant == "mxfp4"
    assert layout.packed_fp4 is True
    assert layout.row_bytes == 128
    assert layout.num_blocks == 8
    assert layout.total_row_bytes() == 136


def test_mxfp4_saves_about_half_the_storage() -> None:
    fp8 = EngramLayout.from_metadata(256, LAYER_IDS, None)
    mxfp4 = _mxfp4_layout()
    assert mxfp4.total_row_bytes() < fp8.total_row_bytes() * 0.52


def test_unsupported_quant_fails() -> None:
    for quant in ("int4", "nvfp4", "", None):
        _expect_error(
            EngramLayout.from_metadata,
            256,
            LAYER_IDS,
            {"engram_embed_quant": quant},
        )


def test_unsupported_block_size_fails() -> None:
    for block in (16, 64, 0, -32):
        _expect_error(
            EngramLayout.from_metadata,
            256,
            LAYER_IDS,
            {"engram_embed_block_size": block},
        )


def test_unsupported_scale_dtype_fails() -> None:
    _expect_error(
        EngramLayout.from_metadata,
        256,
        LAYER_IDS,
        {"engram_embed_scale_dtype": "fp32"},
    )


def test_partial_layer_list_fails() -> None:
    _expect_error(
        EngramLayout.from_metadata,
        256,
        LAYER_IDS,
        {"engram_embed_layers": [17]},
    )


def test_matching_layer_list_is_accepted() -> None:
    layout = EngramLayout.from_metadata(
        256, LAYER_IDS, {"engram_embed_layers": [19, 17]}
    )
    assert layout.packed_fp4 is False


def test_non_dict_quantization_config_fails() -> None:
    for bad_config in ("mxfp4", 32, ["mxfp4"]):
        _expect_error(EngramLayout.from_metadata, 256, LAYER_IDS, bad_config)


def test_non_iterable_layer_list_fails() -> None:
    _expect_error(
        EngramLayout.from_metadata,
        256,
        LAYER_IDS,
        {"engram_embed_layers": 17},
    )


def test_invalid_head_dim_fails() -> None:
    for head_dim in (0, -8, 255, 257):
        _expect_error(EngramLayout.from_metadata, head_dim, LAYER_IDS, None)


def test_bool_head_dim_fails() -> None:
    for head_dim in (True, False):
        _expect_error(EngramLayout.from_metadata, head_dim, LAYER_IDS, None)


def test_bool_block_size_fails() -> None:
    _expect_error(
        EngramLayout.from_metadata,
        256,
        LAYER_IDS,
        {"engram_embed_block_size": True},
    )


def test_head_dim_not_divisible_by_block_fails() -> None:
    _expect_error(EngramLayout.from_metadata, 48, LAYER_IDS, None)


# --- E2M1 decode ----------------------------------------------------------


def test_all_sixteen_codes_match_reference_table() -> None:
    expected = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ]
    got = [decode_e2m1_nibble(code) for code in range(16)]
    assert got == expected


def test_positive_magnitudes_match_lut() -> None:
    for code, value in enumerate(E2M1_VALUES):
        assert decode_e2m1_nibble(code) == value


def test_invalid_nibble_fails() -> None:
    for bad in (-1, 16, 255, 1.5, "3", None):
        _expect_error(decode_e2m1_nibble, bad)


def test_packed_decode_places_even_column_in_low_nibble() -> None:
    assert decode_e2m1_packed([0x91], dim=2) == [0.5, -0.5]


def test_packed_decode_covers_all_bytes() -> None:
    packed = list(range(256))
    decoded = decode_e2m1_packed(packed)
    assert len(decoded) == 512
    for i, byte in enumerate(packed):
        assert decoded[2 * i] == decode_e2m1_nibble(byte & 0xF)
        assert decoded[2 * i + 1] == decode_e2m1_nibble((byte >> 4) & 0xF)


def test_pack_decode_round_trip() -> None:
    values = [0.0, -0.5, 6.0, -6.0, 1.5, -1.5, 3.0, -3.0]
    assert decode_e2m1_packed(pack_e2m1(values), dim=len(values)) == values


def test_pack_rejects_odd_length() -> None:
    _expect_error(pack_e2m1, [0.0, 1.0, 2.0])


def test_pack_rejects_unrepresentable_value() -> None:
    _expect_error(pack_e2m1, [0.0, 0.25])


def test_decode_rejects_short_packed_data() -> None:
    _expect_error(decode_e2m1_packed, [0x00, 0x01], dim=8)


def test_decode_rejects_long_packed_data() -> None:
    _expect_error(decode_e2m1_packed, [0x00] * 5, dim=8)


def test_decode_rejects_non_byte() -> None:
    _expect_error(decode_e2m1_packed, [256])


def test_decode_rejects_non_byte_without_dim() -> None:
    _expect_error(decode_e2m1_packed, [-1])


def test_decode_rejects_bool_byte() -> None:
    _expect_error(decode_e2m1_packed, [True])


def test_decode_rejects_odd_dim() -> None:
    _expect_error(decode_e2m1_packed, [0x00], dim=3)


# --- row-width aware allocation -------------------------------------------


def test_row_bytes_uses_packed_width() -> None:
    assert row_bytes_for(_mxfp4_layout(), 4096) == 4096 * 136
    assert (
        row_bytes_for(EngramLayout.from_metadata(256, LAYER_IDS, None), 4096)
        == 4096 * 264
    )


def test_row_bytes_zero_rows_is_zero() -> None:
    assert row_bytes_for(_mxfp4_layout(), 0) == 0


def test_row_bytes_rejects_negative_rows() -> None:
    _expect_error(row_bytes_for, _mxfp4_layout(), -1)


def test_allocate_within_block_limit() -> None:
    assert allocate_row_buffer(_mxfp4_layout(), 8, max_blocks=8) == 8 * 136


def test_allocate_rejects_block_count_above_limit() -> None:
    _expect_error(allocate_row_buffer, _mxfp4_layout(), 8, max_blocks=4)


def test_allocate_rejects_non_positive_max_blocks() -> None:
    _expect_error(allocate_row_buffer, _mxfp4_layout(), 8, max_blocks=0)


def test_allocate_handles_non_power_of_two_row_width() -> None:
    layout = EngramLayout.from_metadata(
        96, LAYER_IDS, {"engram_embed_quant": "mxfp4"}
    )
    assert layout.row_bytes == 48
    assert layout.num_blocks == 3
    assert allocate_row_buffer(layout, 4, max_blocks=3) == 4 * 51


def test_allocate_rejects_zero_row_width() -> None:
    layout = EngramLayout(
        quant="mxfp4",
        head_dim=0,
        block_size=32,
        scale_dtype="ue8m0",
        packed_fp4=True,
        row_bytes=0,
        num_blocks=0,
    )
    _expect_error(allocate_row_buffer, layout, 4, max_blocks=8)


# --- lookup consistency across storage paths ------------------------------


def _fp8_row(values):
    return [int(v) & 0xFF for v in values]


def test_lookup_matches_between_packed_and_unpacked_tables() -> None:
    head_dim = 256
    fp8 = EngramLayout.from_metadata(head_dim, LAYER_IDS, None)
    mxfp4 = _mxfp4_layout(head_dim)

    values = [decode_e2m1_nibble(c) for c in range(16)] * (head_dim // 16)
    packed_row = pack_e2m1(values)
    fp8_row = _fp8_row([abs(v) for v in values])
    scales = [127] * fp8.num_blocks

    packed_out = lookup_rows(mxfp4, [packed_row], [scales], [0])[0]
    fp8_out = lookup_rows(fp8, [fp8_row], [scales], [0])[0]
    assert len(packed_out) == len(fp8_out) == head_dim
    for a, b in zip(packed_out, fp8_out):
        assert abs(a - b) < 1e-6


def test_lookup_negative_id_returns_zeros() -> None:
    layout = _mxfp4_layout()
    row = pack_e2m1([1.0] * layout.head_dim)
    out = lookup_rows(layout, [row], [[127] * layout.num_blocks], [-1])[0]
    assert out == [0.0] * layout.head_dim


def test_lookup_rejects_out_of_range_id() -> None:
    layout = _mxfp4_layout()
    row = pack_e2m1([1.0] * layout.head_dim)
    _expect_error(lookup_rows, layout, [row], [[127] * layout.num_blocks], [1])


def test_lookup_rejects_mismatched_table_lengths() -> None:
    layout = _mxfp4_layout()
    row = pack_e2m1([1.0] * layout.head_dim)
    _expect_error(lookup_rows, layout, [row], [], [0])


def test_dequantize_rejects_short_packed_row() -> None:
    layout = _mxfp4_layout()
    _expect_error(
        dequantize_row,
        layout,
        [0x00] * (layout.row_bytes - 1),
        [127] * layout.num_blocks,
    )


def test_dequantize_rejects_long_packed_row() -> None:
    layout = _mxfp4_layout()
    _expect_error(
        dequantize_row,
        layout,
        [0x00] * (layout.row_bytes + 1),
        [127] * layout.num_blocks,
    )


def test_dequantize_rejects_block_count_mismatch() -> None:
    layout = _mxfp4_layout()
    row = pack_e2m1([1.0] * layout.head_dim)
    _expect_error(dequantize_row, layout, row, [127] * (layout.num_blocks - 1))


def test_dequantize_rejects_non_byte_scale() -> None:
    layout = _mxfp4_layout()
    row = pack_e2m1([1.0] * layout.head_dim)
    scales = [127] * layout.num_blocks
    scales[0] = 256
    _expect_error(dequantize_row, layout, row, scales)


def test_dequantize_applies_ue8m0_scale_per_block() -> None:
    layout = _mxfp4_layout()
    row = pack_e2m1([1.0] * layout.head_dim)
    scales = [127] * layout.num_blocks
    scales[0] = 128  # 2x for the first block only
    out = dequantize_row(layout, row, scales)
    assert abs(out[0] - 2.0) < 1e-6
    assert abs(out[layout.block_size] - 1.0) < 1e-6


def _collect_tests() -> list[tuple[str, object]]:
    return [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]


def _run_all() -> int:
    tests = _collect_tests()
    failures = []
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures.append((name, exc))
    for name, exc in failures:
        print(f"FAIL {name}: {exc!r}")
    print(f"{len(tests) - len(failures)}/{len(tests)} self-check cases passed")
    return 1 if failures else 0


def test_selfcheck_suite_is_collectable() -> None:
    """Guard: the stdlib-only suite must expose a non-trivial case count.

    This keeps the fallback entry meaningful when pytest is available but the
    heavy dependency set (torch, triton, ...) is not, and it fails loudly if a
    future edit accidentally removes the self-check cases.
    """
    names = [name for name, _ in _collect_tests()]
    assert len(names) >= 40, f"expected >= 40 self-check cases, got {len(names)}"
    assert "test_selfcheck_suite_is_collectable" in names


def test_module_loads_without_sglang_package() -> None:
    """Smoke test: the layout module must load without importing sglang.

    CI environment preparation can fail while installing the full sglang
    dependency chain. This test asserts that the path-based fallback in
    ``_load_engram_mxfp4`` works even when ``sglang`` is not importable, so the
    self-check entry stays usable in a minimal environment.
    """
    import importlib.util

    module_path = os.path.join(
        _PYTHON_DIR, "sglang", "srt", "layers", "engram_mxfp4.py"
    )
    assert os.path.isfile(module_path), f"missing module file: {module_path}"
    spec = importlib.util.spec_from_file_location(
        "engram_mxfp4_smoke", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "EngramLayout")
    assert hasattr(module, "decode_e2m1_nibble")
    assert hasattr(module, "allocate_row_buffer")
    layout = module.EngramLayout.from_metadata(
        256, (17, 19), {"engram_embed_quant": "mxfp4"}
    )
    assert layout.row_bytes == 128
    assert layout.total_row_bytes() == 136


if __name__ == "__main__":
    raise SystemExit(_run_all())
