"""Reward curriculum invariants — the properties GRPO advantage depends on."""

import pytest

from resource_tuner.environment.simulator import simulate_episode
from resource_tuner.policy.actions import Proposal
from resource_tuner.rewards.rewards import (
    MAX_REWARD,
    invalid_proposal_reward,
    overprovision_fraction,
    score_episode,
    stage_a_reward,
    stage_b_reward,
)


def _episode(cpu, mem_mib, true_mem=800.0, true_cpu=1.0, dur=60):
    return simulate_episode(Proposal(cpu=cpu, memory_mib=mem_mib), true_mem, true_cpu, dur)


def test_stage_a_is_binary_on_success():
    fits = stage_a_reward(_episode(1, 1024))
    ooms = stage_a_reward(_episode(1, 512))
    assert fits.total == MAX_REWARD
    assert ooms.total < fits.total
    assert ooms.success == 0.0
    # Stage A must NOT leak waste signal — that's stage B's job.
    assert stage_a_reward(_episode(16, 65536)).total == MAX_REWARD


def test_invalid_proposal_is_the_floor():
    floor = invalid_proposal_reward().total
    assert floor == 0.0
    assert stage_a_reward(_episode(1, 512)).total >= floor


def test_stage_b_prefers_tight_over_padded():
    tight = stage_b_reward(_episode(1, 1024))
    padded = stage_b_reward(_episode(8, 32768))
    assert tight.total > padded.total
    assert padded.waste_penalty < tight.waste_penalty <= 0


def test_stage_b_oom_costs_more_than_any_waste():
    """The PRD's asymmetry: a failed run must score below the most wasteful
    successful run, or the policy learns that failing cheap beats fitting."""
    oom = stage_b_reward(_episode(1, 512))
    most_wasteful_success = stage_b_reward(_episode(16, 65536))
    assert oom.total < most_wasteful_success.total
    assert oom.waste_penalty == 0.0  # waste never counted on failures


def test_stage_b_success_monotone_in_memory_up_to_fit():
    """More memory below the fit line never beats fitting."""
    just_fits = stage_b_reward(_episode(1, 1024))
    for mem in (128, 256, 512):
        assert stage_b_reward(_episode(1, mem)).total < just_fits.total


def test_throttle_is_a_nudge_not_a_failure():
    throttled = stage_b_reward(_episode(0.5, 1024, true_cpu=2.0))
    unthrottled = stage_b_reward(_episode(2, 1024, true_cpu=2.0))
    assert throttled.success > 0  # cpu is compressible: still a success
    assert throttled.throttle_penalty < 0
    assert abs(throttled.throttle_penalty) < 0.1


def test_overprovision_fraction_bounds():
    assert overprovision_fraction(8, 2) == 0.75
    assert overprovision_fraction(2, 8) == 0.0
    assert overprovision_fraction(0, 5) == 0.0
    assert 0.0 <= overprovision_fraction(1e9, 0.001) <= 1.0


def test_unknown_stage_fails_loudly():
    with pytest.raises(ValueError, match="unknown reward stage"):
        score_episode("stage_c", _episode(1, 1024))
