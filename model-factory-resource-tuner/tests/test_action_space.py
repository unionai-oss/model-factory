"""The discretized action space: the decision model's output schema.

This is the contract between "the model chose cell (i, j, k)" and "flyte got a
schedulable `Resources(cpu=..., memory=..., gpu=...)`". A bug here silently
mis-provisions every proposal the arm makes, so the round-trip and the
rounding DIRECTION are both pinned.
"""

import pytest

from resource_tuner.policy.action_space import (
    ACTION_SPACES,
    COARSE_SPACE,
    DEFAULT_SPACE,
    FINE_SPACE,
    ActionSpace,
    get_action_space,
    grid_headroom,
)
from resource_tuner.policy.actions import GPU_VRAM_MIB, Proposal


def test_every_named_space_is_valid_and_registered():
    for name, space in ACTION_SPACES.items():
        assert space.name == name
        assert space.head_sizes[0] > 0 and space.head_sizes[1] > 0 and space.head_sizes[2] > 0
    assert get_action_space("default") is DEFAULT_SPACE
    with pytest.raises(ValueError, match="unknown action space"):
        get_action_space("nope")


def test_grids_must_be_ascending_and_gpu_zero_is_cpu_only():
    with pytest.raises(ValueError, match="ascending"):
        ActionSpace(name="bad", memory_grid_mib=(1024, 512))
    with pytest.raises(ValueError, match="ascending"):
        ActionSpace(name="bad", cpu_grid=(4, 1))
    with pytest.raises(ValueError, match="CPU-only"):
        ActionSpace(name="bad", gpu_options=(("T4", 1), None))
    with pytest.raises(ValueError, match="empty dimension"):
        ActionSpace(name="bad", memory_grid_mib=())


# ── rounding direction: the expensive asymmetry ─────────────────────────


def test_memory_rounds_UP_to_the_covering_bucket():
    # Rounding down would under-provision and OOM the task. This is the single
    # most important property in the module.
    space = DEFAULT_SPACE
    i = space.memory_index(1025)
    assert space.memory_grid_mib[i] == 2048
    assert space.memory_grid_mib[i] >= 1025


def test_an_exact_grid_value_takes_its_own_bucket_not_the_next():
    space = DEFAULT_SPACE
    i = space.memory_index(1024)
    assert space.memory_grid_mib[i] == 1024


def test_cpu_rounds_up_too():
    space = DEFAULT_SPACE
    assert space.cpu_grid[space.cpu_index(2.1)] == 4
    assert space.cpu_grid[space.cpu_index(2.0)] == 2


def test_above_the_grid_clamps_to_the_largest_bucket():
    space = DEFAULT_SPACE
    assert space.memory_index(10**9) == len(space.memory_grid_mib) - 1
    assert space.cpu_index(10**6) == len(space.cpu_grid) - 1


# ── GPU selection ───────────────────────────────────────────────────────


def test_zero_gpu_memory_is_the_cpu_only_choice():
    assert DEFAULT_SPACE.gpu_index(0) == 0
    assert DEFAULT_SPACE.gpu_index(None) == 0
    assert DEFAULT_SPACE.gpu_index(-5) == 0


def test_gpu_picks_the_cheapest_card_that_fits():
    space = DEFAULT_SPACE
    # T4 has 16Gi; something needing 8Gi should take the T4, not an L40S.
    idx = space.gpu_index(8 * 1024)
    option = space.gpu_options[idx]
    assert option is not None and option[0] == "T4"

    # 20Gi does not fit a T4; the next cheapest that does is the L4 (24Gi).
    idx = space.gpu_index(20 * 1024)
    assert space.gpu_options[idx][0] == "L4"


def test_a_requirement_bigger_than_every_card_clamps_to_the_largest():
    # Not to CPU-only: an under-provisioned GPU request kills the task, which
    # is the expensive direction.
    space = DEFAULT_SPACE
    idx = space.gpu_index(10**7)
    option = space.gpu_options[idx]
    assert option is not None
    best = max(GPU_VRAM_MIB[o[0]] * o[1] for o in space.gpu_options if o)
    assert GPU_VRAM_MIB[option[0]] * option[1] == best


# ── round trip to flyte Resources ───────────────────────────────────────


def test_decode_produces_a_proposal_that_maps_to_flyte_kwargs():
    space = DEFAULT_SPACE
    proposal = space.decode(space.memory_index(3000), space.cpu_index(3), space.gpu_index(0))
    assert isinstance(proposal, Proposal)
    kwargs = proposal.to_kwargs()
    # This is what reaches flyte.Resources(...).
    assert kwargs["memory"] == "4Gi"
    assert kwargs["cpu"] == 4
    assert "gpu" not in kwargs  # CPU-only choice must not request a GPU


def test_a_gpu_choice_renders_as_a_typed_flyte_gpu_request():
    space = DEFAULT_SPACE
    proposal = space.decode(5, 2, space.gpu_index(8 * 1024))
    kwargs = proposal.to_kwargs()
    assert kwargs["gpu"] == "T4:1"


def test_encode_decode_round_trips_through_head_indices():
    space = DEFAULT_SPACE
    original = Proposal(cpu=4, memory_mib=8192, gpu=1, gpu_type="L4")
    indices = space.encode_proposal(original)
    back = space.decode(*indices)
    assert back.memory_mib == original.memory_mib
    assert back.cpu == original.cpu
    assert back.gpu == original.gpu
    assert back.gpu_type == original.gpu_type


def test_labels_are_the_cheapest_cell_that_covers_the_truth():
    space = DEFAULT_SPACE
    mem_i, cpu_i, gpu_i = space.encode_labels(1500, 3.2, 0)
    assert space.memory_grid_mib[mem_i] == 2048  # covers 1500
    assert space.cpu_grid[cpu_i] == 4  # covers 3.2
    assert space.gpu_options[gpu_i] is None  # a CPU task


def test_decode_clamps_out_of_range_indices_instead_of_raising():
    # A corrupt checkpoint should still serve something schedulable rather
    # than raising inside the tune service.
    space = DEFAULT_SPACE
    low = space.decode(-10, -10, -10)
    high = space.decode(999, 999, 999)
    assert low.memory_mib == space.memory_grid_mib[0]
    assert high.memory_mib == space.memory_grid_mib[-1]


# ── the hyperparameter axis ─────────────────────────────────────────────


def test_coarser_grids_mean_fewer_classes_and_more_forced_waste():
    assert COARSE_SPACE.n_joint < DEFAULT_SPACE.n_joint < FINE_SPACE.n_joint
    true = [300.0, 1100.0, 3000.0, 5000.0, 20000.0]
    coarse = grid_headroom(COARSE_SPACE, true)
    fine = grid_headroom(FINE_SPACE, true)
    # A finer grid can never force MORE rounding waste than a coarser one.
    assert fine <= coarse


def test_grid_headroom_is_zero_on_exact_grid_values():
    space = DEFAULT_SPACE
    assert grid_headroom(space, list(space.memory_grid_mib[:4])) == pytest.approx(0.0)


def test_grid_headroom_ignores_degenerate_inputs():
    assert grid_headroom(DEFAULT_SPACE, []) == 0.0
    assert grid_headroom(DEFAULT_SPACE, [0.0, -1.0]) == 0.0


def test_space_serializes_round_trip():
    # A checkpoint is undecodable without the grid it was trained on, so the
    # manifest has to carry it losslessly.
    for space in (DEFAULT_SPACE, COARSE_SPACE, FINE_SPACE):
        back = ActionSpace.from_dict(space.to_dict())
        assert back.name == space.name
        assert back.memory_grid_mib == space.memory_grid_mib
        assert back.cpu_grid == space.cpu_grid
        assert back.gpu_options == space.gpu_options
