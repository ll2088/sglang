from dataclasses import replace

import pytest

from ..config import (
    FwdHeuristicInputs,
    Sm90Registers,
    Sm100Registers,
    num_splits_heuristic,
    select_fwd_config,
    select_registers,
    validate_fwd_config,
    validate_registers,
)

_BASE_INPUTS = FwdHeuristicInputs(
    device_arch=100,
    num_sms=132,
    dtype="torch.bfloat16",
    head_dim=64,
    head_dim_v=64,
    num_heads=16,
    num_heads_kv=16,
    batch_size=2,
    total_q=8192,
    total_k=8192,
    max_seqlen_q=4096,
    max_seqlen_k=4096,
    requested_num_splits=1,
)


def make_inputs(**changes) -> FwdHeuristicInputs:
    return _BASE_INPUTS._replace(**changes)


@pytest.mark.parametrize("device_arch", [80, 90, 100, 103, 110, 120])
def test_every_architecture_selects_a_valid_default(device_arch):
    config = select_fwd_config(make_inputs(device_arch=device_arch))
    validate_fwd_config(config, make_inputs(device_arch=device_arch))


def test_2cta_boundary_follows_packed_q_tiles():
    one_cta = make_inputs(
        dtype="torch.float16", head_dim=128, head_dim_v=128, max_seqlen_q=256
    )

    assert not select_fwd_config(one_cta).use_2cta_instrs
    assert select_fwd_config(one_cta._replace(max_seqlen_q=257)).use_2cta_instrs


def test_explicit_2cta_is_valid_when_the_environment_disables_the_default():
    inputs = make_inputs(head_dim=128, head_dim_v=128, disable_2cta=True)
    config = select_fwd_config(inputs)

    assert not config.use_2cta_instrs
    validate_fwd_config(replace(config, use_2cta_instrs=True), inputs)


def test_varlen_q_disables_tma_output_and_static_persistence():
    inputs = make_inputs(has_cu_seqlens_q=True, batch_size=4, max_seqlen_q=512)
    config = select_fwd_config(inputs)

    assert not config.is_static_persistent
    assert not config.use_tma_o
    with pytest.raises(ValueError, match="TMA O"):
        validate_fwd_config(replace(config, use_tma_o=True), inputs)


def test_semaphore_disables_clc_and_static_persistence():
    inputs = make_inputs(
        has_caller_scheduler_metadata=True,
        has_tile_count_semaphore=True,
        requested_use_clc_scheduler=True,
    )
    config = select_fwd_config(inputs)

    assert not config.use_clc_scheduler
    assert not config.is_static_persistent
    with pytest.raises(ValueError, match="CLC"):
        validate_fwd_config(replace(config, use_clc_scheduler=True), inputs)
    with pytest.raises(ValueError, match="Static persistent"):
        validate_fwd_config(replace(config, is_static_persistent=True), inputs)


def test_clc_excludes_static_persistence():
    inputs = make_inputs(has_caller_scheduler_metadata=True, device_arch=110)
    config = select_fwd_config(inputs)

    assert config.use_clc_scheduler
    assert not config.is_static_persistent
    with pytest.raises(ValueError, match="Static persistent scheduling"):
        validate_fwd_config(replace(config, is_static_persistent=True), inputs)


def test_explicit_split_count_is_kept_exactly():
    config = select_fwd_config(
        make_inputs(head_dim=128, head_dim_v=128, requested_num_splits=7)
    )

    assert config.num_splits == 7
    with pytest.raises(ValueError, match="num_splits"):
        validate_fwd_config(replace(config, num_splits=0), _BASE_INPUTS)


def test_registers_are_arch_specific():
    sm100 = select_registers(_BASE_INPUTS, 128, 128, 2, False)
    assert isinstance(sm100, Sm100Registers)
    validate_registers(sm100, _BASE_INPUTS, 2)

    sm90_inputs = make_inputs(device_arch=90)
    sm90 = select_registers(sm90_inputs, 128, 128, 2, False)
    assert isinstance(sm90, Sm90Registers)
    validate_registers(sm90, sm90_inputs, 2)

    sm80_inputs = make_inputs(device_arch=80)
    sm80 = select_registers(sm80_inputs, 128, 128, 2, False)
    assert sm80 is None
    validate_registers(sm80, sm80_inputs, 2)


def test_register_validation_rejects_out_of_budget():
    with pytest.raises(ValueError, match="register budget exceeded"):
        validate_registers(
            Sm100Registers(softmax=256, correction=252, other=96), _BASE_INPUTS, 2
        )
    with pytest.raises(ValueError, match="softmax registers must be"):
        validate_registers(
            Sm100Registers(softmax=64, correction=64, other=64), _BASE_INPUTS, 2
        )
    with pytest.raises(ValueError, match="producer registers"):
        validate_registers(
            Sm90Registers(mma=128, producer=200), make_inputs(device_arch=90), 2
        )


def test_num_splits_heuristic_only_splits_for_parallelism():
    assert num_splits_heuristic(_BASE_INPUTS, 132) == 1
    low_parallelism = make_inputs(total_q=128, max_seqlen_k=8192)
    assert num_splits_heuristic(low_parallelism, 132) > 1
