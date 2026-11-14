"""Unit tests for MXFP4-packed Engram layout parsing, E2M1 decode, and allocation."""

from __future__ import annotations

import pytest

from sglang.srt.layers.engram_mxfp4 import (
    E2M1_VALUES,
    EngramLayout,
    EngramLayoutError,
    allocate_row_buffer,
    decode_e2m1_nibble,
    decode_e2m1_packed,
    pack_e2m1,
    row_bytes_for,
)


LAYER_IDS = (17, 19)


def _mxfp4_layout(head_dim: int = 256) -> EngramLayout:
    return EngramLayout.from_metadata(
        head_dim,
        LAYER_IDS,
        {"engram_embed_quant": "mxfp4"},
    )


# --- layout parsing -------------------------------------------------------


def test_default_layout_is_fp8():
    layout = EngramLayout.from_metadata(256, LAYER_IDS, None)
    assert layout.quant == "fp8"
    assert layout.packed_fp4 is False
    assert layout.row_bytes == 256
    assert layout.num_blocks == 8
    assert layout.total_row_bytes() == 264


def test_mxfp4_layout_halves_row_width():
    layout = _mxfp4_layout()
    assert layout.quant == "mxfp4"
    assert layout.packed_fp4 is True
    assert layout.row_bytes == 128
    assert layout.num_blocks == 8
    assert layout.total_row_bytes() == 136


def test_mxfp4_saves_about_half_the_storage():
    fp8 = EngramLayout.from_metadata(256, LAYER_IDS, None)
    mxfp4 = _mxfp4_layout()
    assert mxfp4.total_row_bytes() < fp8.total_row_bytes() * 0.52


@pytest.mark.parametrize("quant", ["int4", "nvfp4", "", None])
def test_unsupported_quant_fails(quant):
    with pytest.raises(EngramLayoutError):
        EngramLayout.from_metadata(256, LAYER_IDS, {"engram_embed_quant": quant})


@pytest.mark.parametrize("block", [16, 64, 0, -32])
def test_unsupported_block_size_fails(block):
    with pytest.raises(EngramLayoutError):
        EngramLayout.from_metadata(
            256, LAYER_IDS, {"engram_embed_block_size": block}
        )


def test_unsupported_scale_dtype_fails():
    with pytest.raises(EngramLayoutError):
        EngramLayout.from_metadata(
            256, LAYER_IDS, {"engram_embed_scale_dtype": "fp32"}
        )


def test_partial_layer_list_fails():
    with pytest.raises(EngramLayoutError):
        EngramLayout.from_metadata(
            256, LAYER_IDS, {"engram_embed_layers": [17]}
        )


def test_matching_layer_list_is_accepted():
    layout = EngramLayout.from_metadata(
        256, LAYER_IDS, {"engram_embed_layers": [19, 17]}
    )
    assert layout.packed_fp4 is False


@pytest.mark.parametrize("head_dim", [0, -8, 255, 257])
def test_invalid_head_dim_fails(head_dim):
    with pytest.raises(EngramLayoutError):
        EngramLayout.from_metadata(head_dim, LAYER_IDS, None)


def test_head_dim_not_divisible_by_block_fails():
    with pytest.raises(EngramLayoutError):
        EngramLayout.from_metadata(48, LAYER_IDS, None)


# --- E2M1 decode ----------------------------------------------------------


def test_all_sixteen_codes_match_reference_table():
    expected = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ]
    got = [decode_e2m1_nibble(code) for code in range(16)]
    assert got == expected


def test_positive_magnitudes_match_lut():
    for code, value in enumerate(E2M1_VALUES):
        assert decode_e2m1_nibble(code) == value


@pytest.mark.parametrize("bad", [-1, 16, 255, 1.5, "3", None])
def test_invalid_nibble_fails(bad):
    with pytest.raises(EngramLayoutError):
        decode_e2m1_nibble(bad)


def test_packed_decode_places_even_column_in_low_nibble():
    # low nibble 0x1 -> 0.5, high nibble 0x9 -> -0.5
    assert decode_e2m1_packed([0x91], dim=2) == [0.5, -0.5]


def test_packed_decode_covers_all_bytes():
    packed = list(range(256))
    decoded = decode_e2m1_packed(packed)
    assert len(decoded) == 512
    for i, byte in enumerate(packed):
        assert decoded[2 * i] == decode_e2m1_nibble(byte & 0xF)
        assert decoded[2 * i + 1] == decode_e2m1_nibble((byte >> 4) & 0xF)


def test_pack_decode_round_trip():
    values = [0.0, -0.5, 6.0, -6.0, 1.5, -1.5, 3.0, -3.0]
    assert decode_e2m1_packed(pack_e2m1(values), dim=len(values)) == values


def test_pack_rejects_odd_length():
    with pytest.raises(EngramLayoutError):
        pack_e2m1([0.0, 1.0, 2.0])


def test_pack_rejects_unrepresentable_value():
    with pytest.raises(EngramLayoutError):
        pack_e2m1([0.0, 0.25])


def test_decode_rejects_short_packed_data():
    with pytest.raises(EngramLayoutError):
        decode_e2m1_packed([0x00, 0x01], dim=8)


def test_decode_rejects_long_packed_data():
    with pytest.raises(EngramLayoutError):
        decode_e2m1_packed([0x00] * 5, dim=8)


def test_decode_rejects_non_byte():
    with pytest.raises(EngramLayoutError):
        decode_e2m1_packed([256])


def test_decode_rejects_odd_dim():
    with pytest.raises(EngramLayoutError):
        decode_e2m1_packed([0x00], dim=3)


# --- row-width aware allocation -------------------------------------------


def test_row_bytes_uses_packed_width():
    assert row_bytes_for(_mxfp4_layout(), 4096) == 4096 * 136
    assert row_bytes_for(EngramLayout.from_metadata(256, LAYER_IDS, None), 4096) == 4096 * 264


def test_row_bytes_zero_rows_is_zero():
    assert row_bytes_for(_mxfp4_layout(), 0) == 0


def test_row_bytes_rejects_negative_rows():
    with pytest.raises(EngramLayoutError):
        row_bytes_for(_mxfp4_layout(), -1)


def test_allocate_within_block_limit():
    assert allocate_row_buffer(_mxfp4_layout(), 8, max_blocks=8) == 8 * 136


def test_allocate_rejects_block_count_above_limit():
    with pytest.raises(EngramLayoutError):
        allocate_row_buffer(_mxfp4_layout(), 8, max_blocks=4)


def test_allocate_rejects_non_positive_max_blocks():
    with pytest.raises(EngramLayoutError):
        allocate_row_buffer(_mxfp4_layout(), 8, max_blocks=0)


def test_allocate_handles_non_power_of_two_row_width():
    # head_dim 96 -> 3 blocks, row_bytes 48; not a power of two.
    layout = EngramLayout.from_metadata(
        96, LAYER_IDS, {"engram_embed_quant": "mxfp4"}
    )
    assert layout.row_bytes == 48
    assert layout.num_blocks == 3
    assert allocate_row_buffer(layout, 4, max_blocks=3) == 4 * 51


def test_allocate_rejects_zero_row_width():
    layout = EngramLayout(
        quant="mxfp4",
        head_dim=0,
        block_size=32,
        scale_dtype="ue8m0",
        packed_fp4=True,
        row_bytes=0,
        num_blocks=0,
    )
    with pytest.raises(EngramLayoutError):
        allocate_row_buffer(layout, 4, max_blocks=8)
