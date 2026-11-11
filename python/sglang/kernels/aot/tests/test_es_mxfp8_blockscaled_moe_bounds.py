import sys

import pytest

# torch is optional at collection time: without it (or without a GPU) the module
# must still be collectible and report skips instead of import errors.
torch = pytest.importorskip("torch", reason="torch is not installed")


def _ops():
    sgl_kernel = pytest.importorskip(
        "sgl_kernel", reason="sgl_kernel extension is not built"
    )
    mm = getattr(sgl_kernel, "es_sm100_mxfp8_blockscaled_grouped_mm", None)
    quant = getattr(sgl_kernel, "es_sm100_mxfp8_blockscaled_grouped_quant", None)
    if mm is None or quant is None:
        pytest.skip("sgl_kernel was built without the sm100 blockscaled MoE ops")
    return mm, quant


def is_sm100_supported(device=None) -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        major = torch.cuda.get_device_capability(device)[0]
    except Exception:
        return False
    return (major == 10) and (
        torch.version.cuda is not None and torch.version.cuda >= "12.8"
    )


if not is_sm100_supported():
    pytest.skip(
        "bounds tests are only supported on sm100",
        allow_module_level=True,
    )


def align(val: int, alignment: int = 128) -> int:
    return int((val + alignment - 1) // alignment * alignment)


def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


@pytest.mark.parametrize("num_experts", [1, 3, 7])
@pytest.mark.parametrize("out_dtype", [torch.half, torch.bfloat16])
@pytest.mark.parametrize("m_sizes", [[1], [1, 1, 127, 129, 255], [128, 129, 256]])
def test_es_mxfp8_blockscaled_moe_tail_bounds(num_experts, out_dtype, m_sizes):
    """Tail M rows and SF blocks must not spill into sentinel regions."""
    es_sm100_mxfp8_blockscaled_grouped_mm, es_sm100_mxfp8_blockscaled_grouped_quant = _ops()
    device = "cuda"
    alignment = 128
    n_g = alignment
    k_g = alignment

    expert_offset = 0
    expert_offsets = []
    aux_expert_offset = 0
    aux_expert_offsets = []
    a_blockscale_offset = 0
    a_blockscale_offsets = []
    b_blockscale_offsets = []
    problem_sizes = []
    aux_problem_sizes = []
    a_list = []
    b_list = []
    ref_d_list = []

    for g in range(num_experts):
        m_g = m_sizes[g % len(m_sizes)]
        expert_offsets.append(expert_offset)
        expert_offset += m_g
        aux_expert_offsets.append(aux_expert_offset)
        aux_expert_offset += n_g
        a_blockscale_offsets.append(a_blockscale_offset)
        a_blockscale_offset += align(m_g, 128)
        b_blockscale_offsets.append(g * n_g)
        problem_sizes.append([m_g, n_g, k_g])
        aux_problem_sizes.append([n_g, m_g, k_g])

        a = torch.normal(0.0, std=1.0, size=(m_g, k_g), device=device, dtype=out_dtype)
        b = torch.normal(0.0, std=1.0, size=(n_g, k_g), device=device, dtype=out_dtype)
        a_list.append(a)
        b_list.append(b)
        ref_d_list.append(a @ b.T)

    a = torch.concat(a_list, dim=0)
    b = torch.concat(b_list, dim=0)

    _problem_sizes = torch.tensor(problem_sizes).to(device=device, dtype=torch.int32)
    _aux_problem_sizes = torch.tensor(aux_problem_sizes).to(device=device, dtype=torch.int32)
    _expert_offsets = torch.tensor(expert_offsets).to(device=device, dtype=torch.int32)
    _aux_expert_offsets = torch.tensor(aux_expert_offsets).to(device=device, dtype=torch.int32)
    _a_blockscale_offsets = torch.tensor(a_blockscale_offsets).to(device=device, dtype=torch.int32)
    _b_blockscale_offsets = torch.tensor(b_blockscale_offsets).to(device=device, dtype=torch.int32)

    a_quant = torch.zeros_like(a, dtype=torch.float8_e4m3fn, device=device)
    a_sf_rows = a_blockscale_offset + 128
    a_scale_factor = torch.zeros((a_sf_rows, k_g // 32), dtype=torch.uint8, device=device)
    a_sf_sentinel = a_scale_factor[a_blockscale_offset:].clone()

    b_quant = torch.zeros_like(b, dtype=torch.float8_e4m3fn, device=device)
    b_scale_factor = torch.zeros((num_experts, n_g, k_g // 32), dtype=torch.uint8, device=device)

    es_sm100_mxfp8_blockscaled_grouped_quant(
        a, _problem_sizes, _expert_offsets, _a_blockscale_offsets, a_quant, a_scale_factor
    )
    es_sm100_mxfp8_blockscaled_grouped_quant(
        b, _aux_problem_sizes, _aux_expert_offsets, _b_blockscale_offsets, b_quant, b_scale_factor
    )

    assert torch.equal(a_scale_factor[a_blockscale_offset:], a_sf_sentinel), (
        "scale-factor store wrote past the allocated SF rows"
    )

    b_quant = b_quant.view(num_experts, n_g, k_g).transpose(1, 2)
    b_scale_factor = b_scale_factor.view(num_experts, n_g, k_g // 32).transpose(1, 2)

    d = torch.zeros((expert_offset + 1, n_g), device=device, dtype=out_dtype)
    d_sentinel = d[expert_offset:].clone()
    es_sm100_mxfp8_blockscaled_grouped_mm(
        d, a_quant, b_quant, a_scale_factor, b_scale_factor, _problem_sizes, _expert_offsets, _a_blockscale_offsets
    )

    assert torch.equal(d[expert_offset:], d_sentinel), "GEMM wrote past the last token row"

    for g in range(num_experts):
        baseline = ref_d_list[g]
        actual = d[expert_offsets[g] : (expert_offsets[g] + problem_sizes[g][0])]
        diff = calc_diff(actual, baseline)
        assert diff < 0.001, f"expert {g} m={baseline.shape[0]} diff={diff:.5f}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
