import pathlib

import pytest

_TESTS_DIR = pathlib.Path(__file__).resolve().parent
_QUANT_TEST_FILES = [
    "test_per_token_group_quant_8bit.py",
    "test_es_mxfp8_blockscaled_moe.py",
    "test_fp8_blockwise_moe.py",
    "test_int8_gemm.py",
]


@pytest.mark.parametrize("filename", _QUANT_TEST_FILES)
def test_quant_tests_do_not_use_mem_get_info(filename):
    """Guard against regressing to device-wide memory measurement in quant tests."""
    path = _TESTS_DIR / filename
    assert path.is_file(), f"missing test file: {path}"
    source = path.read_text(encoding="utf-8")
    assert "mem_get_info" not in source, (
        f"{filename} must use allocator peak stats instead of torch.cuda.mem_get_info"
    )
